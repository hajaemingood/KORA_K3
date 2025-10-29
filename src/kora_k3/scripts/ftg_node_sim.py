#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


class FollowTheGapNode:
    """Follow-The-Gap 알고리즘을 f1tenth_simulator에서 사용 가능하도록 구성한 ROS 노드."""

    def __init__(self) -> None:
        # parameters
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.drive_topic = rospy.get_param("~drive_topic", "/drive")
        self.max_scan_range = rospy.get_param("~max_scan_range", 12.0)
        self.conv_window = rospy.get_param("~smoothing_window", 5)
        self.bubble_points = rospy.get_param("~bubble_num_points", 22)
        self.obstacle_threshold = rospy.get_param("~obstacle_threshold", 1.0)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.34)
        self.max_speed = rospy.get_param("~max_speed", 3.0)
        self.min_speed = rospy.get_param("~min_speed", 1.0)
        self.speed_drop_steer = rospy.get_param("~speed_drop_steer_angle", 0.25)
        self.front_angle_window = rospy.get_param("~front_angle_window", 0.6)
        self.forward_angle_limit = rospy.get_param("~forward_angle_limit", 0.9)
        self.front_brake_distance = rospy.get_param("~front_brake_distance", 1.8)
        self.front_stop_distance = rospy.get_param("~front_stop_distance", 0.8)
        self.collision_brake_speed = rospy.get_param("~collision_brake_speed", 0.5)
        self.creep_speed = rospy.get_param("~creep_speed", 0.2)
        self.gap_centering_weight = rospy.get_param("~gap_centering_weight", 0.3)
        self.gap_distance_weight = rospy.get_param("~gap_distance_weight", 1.0)
        self.gap_width_weight = rospy.get_param("~gap_width_weight", 0.25)
        self.gap_alignment_weight = rospy.get_param("~gap_alignment_weight", 1.6)
        self.path_alignment_weight = rospy.get_param("~path_alignment_weight", 1.4)
        self.use_path_blending = rospy.get_param("~use_path_blending", True)
        self.path_drive_topic = rospy.get_param("~path_drive_topic", "/pure_pursuit/drive")
        self.path_timeout = rospy.get_param("~path_command_timeout", 0.5)
        self.blend_activate_distance = rospy.get_param("~blend_activate_distance", 2.0)
        self.blend_full_distance = rospy.get_param("~blend_full_distance", 0.9)
        self.turn_blend_scale = rospy.get_param("~turn_blend_scale", 0.15)
        self.path_heading_window = rospy.get_param("~path_heading_window", 0.8)

        self.angle_increment = None
        self.angle_min = None
        self.front_indices = None
        self.forward_mask = None
        self.scan_angles = None
        self.last_path_cmd: Optional[AckermannDriveStamped] = None
        self.last_path_time = rospy.Time(0)

        # pub / sub
        self.drive_pub = rospy.Publisher(
            self.drive_topic, AckermannDriveStamped, queue_size=1
        )
        rospy.Subscriber(
            self.scan_topic, LaserScan, self.scan_callback, queue_size=1
        )
        if self.use_path_blending:
            rospy.Subscriber(
                self.path_drive_topic, AckermannDriveStamped, self.path_callback, queue_size=1
            )

    def scan_callback(self, scan: LaserScan) -> None:
        if np.isnan(scan.angle_increment) or scan.angle_increment == 0.0:
            rospy.logwarn_throttle(5.0, "유효하지 않은 angle_increment 값을 받았습니다.")
            return

        if self.angle_increment is None:
            self.angle_increment = scan.angle_increment
            self.angle_min = scan.angle_min
            self._update_front_indices(len(scan.ranges))
            self._update_forward_mask(len(scan.ranges))
            self._update_scan_angles(len(scan.ranges))
        else:
            # 드물게 각도 증분이 변하면 최신 값을 반영
            if not math.isclose(self.angle_increment, scan.angle_increment):
                self.angle_increment = scan.angle_increment
                self._update_front_indices(len(scan.ranges))
                self._update_forward_mask(len(scan.ranges))
                self._update_scan_angles(len(scan.ranges))
            if not math.isclose(self.angle_min or 0.0, scan.angle_min):
                self.angle_min = scan.angle_min
                self._update_front_indices(len(scan.ranges))
                self._update_forward_mask(len(scan.ranges))
                self._update_scan_angles(len(scan.ranges))

        ranges = np.array(scan.ranges, dtype=np.float32)

        processed = self._preprocess_scan(ranges)
        front_distance = self._front_distance(processed)
        constrained = self._apply_forward_window(processed, front_distance)
        bubble_masked = self._mask_bubble(constrained)
        gaps = self._extract_gaps(bubble_masked)
        if not gaps:
            rospy.logwarn_throttle(2.0, "사용 가능한 gap을 찾지 못했습니다. 차량 정지.")
            self.publish_drive(0.0, 0.0)
            return

        start, end = self._choose_gap(gaps, bubble_masked, front_distance)
        best_index = self._select_best_point(bubble_masked, start, end, front_distance)
        steering = self._index_to_steering(best_index)
        speed = self._compute_speed(abs(steering), front_distance)

        steering, speed = self._blend_with_path_command(steering, speed, front_distance)

        self.publish_drive(speed, steering)

    def _preprocess_scan(self, ranges: np.ndarray) -> np.ndarray:
        ranges = np.nan_to_num(ranges, nan=self.max_scan_range, posinf=self.max_scan_range)
        ranges = np.clip(ranges, 0.0, self.max_scan_range)

        if self.conv_window > 1:
            kernel = np.ones(self.conv_window) / float(self.conv_window)
            ranges = np.convolve(ranges, kernel, mode="same")
        return ranges

    def _mask_bubble(self, processed: np.ndarray) -> np.ndarray:
        closest_index = np.argmin(processed)
        masked = processed.copy()
        half_window = max(self.bubble_points // 2, 1)
        start = max(closest_index - half_window, 0)
        end = min(closest_index + half_window, processed.size - 1)
        masked[start : end + 1] = 0.0
        return masked

    def _extract_gaps(self, scan: np.ndarray) -> List[Tuple[int, int]]:
        gaps: List[Tuple[int, int]] = []
        current_start = None

        for idx, distance in enumerate(scan):
            if distance > self.obstacle_threshold:
                if current_start is None:
                    current_start = idx
            else:
                if current_start is not None:
                    gaps.append((current_start, idx - 1))
                current_start = None

        if current_start is not None:
            gaps.append((current_start, scan.size - 1))
        return gaps

    def _choose_gap(
        self, gaps: List[Tuple[int, int]], scan: np.ndarray, front_distance: float
    ) -> Tuple[int, int]:
        best_gap = gaps[0]
        best_score = float("-inf")
        for start, end in gaps:
            if end <= start:
                continue
            segment = scan[start : end + 1]
            if segment.size == 0:
                continue
            distance_score = self.gap_distance_weight * float(np.max(segment))
            width_rad = (end - start) * (self.angle_increment or 0.0)
            width_score = self.gap_width_weight * width_rad
            center_idx = (start + end) // 2
            center_angle = self._index_to_angle(center_idx)
            gap_alignment = self.gap_alignment_weight
            path_alignment = self.path_alignment_weight
            if front_distance <= self.front_stop_distance:
                gap_alignment *= 0.25
                path_alignment *= 0.25
            elif front_distance <= self.front_brake_distance:
                gap_alignment *= 0.6
                path_alignment *= 0.5
            alignment_penalty = gap_alignment * abs(center_angle)
            if (
                self.use_path_blending
                and self.last_path_cmd is not None
                and (rospy.Time.now() - self.last_path_time).to_sec() <= self.path_timeout
            ):
                desired_angle = float(self.last_path_cmd.drive.steering_angle)
                alignment_penalty += path_alignment * abs(center_angle - desired_angle)
            score = distance_score + width_score - alignment_penalty
            if score > best_score:
                best_score = score
                best_gap = (start, end)
        return best_gap

    def _select_best_point(
        self, scan: np.ndarray, start: int, end: int, front_distance: float
    ) -> int:
        segment = scan[start : end + 1]
        if segment.size == 0:
            return start
        best_rel_index = np.argmax(segment)
        best_index = start + int(best_rel_index)
        mid_index = (start + end) // 2
        weight = np.clip(self.gap_centering_weight, 0.0, 1.0)
        if front_distance <= self.front_stop_distance:
            weight = max(0.0, weight - 0.5)
        elif front_distance <= self.front_brake_distance:
            weight = max(0.0, weight - 0.25)
        return int(weight * mid_index + (1.0 - weight) * best_index)

    def _index_to_steering(self, index: int) -> float:
        if self.angle_increment is None:
            return 0.0
        base_angle = self._index_to_angle(index)
        return max(min(base_angle, self.max_steering_angle), -self.max_steering_angle)

    def _compute_speed(self, abs_steer: float, front_distance: float) -> float:
        if abs_steer >= self.speed_drop_steer:
            base_speed = self.min_speed
        else:
            ratio = abs_steer / self.speed_drop_steer if self.speed_drop_steer > 0 else 0.0
            base_speed = self.max_speed - (self.max_speed - self.min_speed) * ratio
            base_speed = max(self.min_speed, min(self.max_speed, base_speed))

        if front_distance < self.front_stop_distance:
            return self.creep_speed
        if front_distance < self.front_brake_distance:
            return min(base_speed, max(self.creep_speed, self.collision_brake_speed))
        return base_speed

    def _front_distance(self, ranges: np.ndarray) -> float:
        # widen detection to entire forward mask if available
        if self.forward_mask is not None and np.any(self.forward_mask):
            return float(np.min(ranges[self.forward_mask]))
        if self.front_indices is not None and self.front_indices.size > 0:
            return float(np.min(ranges[self.front_indices]))
        return float(np.min(ranges))

    def _update_front_indices(self, total_points: int) -> None:
        if self.angle_increment is None or self.angle_min is None:
            self.front_indices = None
            return
        angles = (self.angle_min + np.arange(total_points) * self.angle_increment).astype(np.float32)
        mask = np.abs(angles) <= self.front_angle_window
        self.front_indices = np.where(mask)[0]

    def _update_forward_mask(self, total_points: int) -> None:
        if self.angle_increment is None or self.angle_min is None:
            self.forward_mask = None
            self.scan_angles = None
            return
        angles = (self.angle_min + np.arange(total_points) * self.angle_increment).astype(np.float32)
        self.forward_mask = np.abs(angles) <= self.forward_angle_limit
        self.scan_angles = angles

    def _update_scan_angles(self, total_points: int) -> None:
        if self.angle_increment is None or self.angle_min is None:
            self.scan_angles = None
            return
        self.scan_angles = (self.angle_min + np.arange(total_points) * self.angle_increment).astype(np.float32)

    def _apply_forward_window(self, ranges: np.ndarray, front_distance: float) -> np.ndarray:
        constrained = ranges.copy()
        if self.forward_mask is not None:
            constrained[~self.forward_mask] = 0.0

        if (
            self.use_path_blending
            and self.last_path_cmd is not None
            and self.scan_angles is not None
            and (rospy.Time.now() - self.last_path_time).to_sec() <= self.path_timeout
        ):
            desired_angle = float(self.last_path_cmd.drive.steering_angle)
            if front_distance <= self.front_stop_distance:
                window = max(self.path_heading_window, 1.2)
            elif front_distance <= self.front_brake_distance:
                window = max(self.path_heading_window, 1.0)
            else:
                window = self.path_heading_window
            diffs = self._angle_difference(self.scan_angles, desired_angle)
            constrained[np.abs(diffs) > window] = 0.0
        return constrained

    def _index_to_angle(self, index: int) -> float:
        if self.angle_increment is None:
            return 0.0
        return (self.angle_min or 0.0) + index * self.angle_increment

    @staticmethod
    def _angle_difference(angles: np.ndarray, reference: float) -> np.ndarray:
        diff = angles - reference
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def path_callback(self, msg: AckermannDriveStamped) -> None:
        self.last_path_cmd = msg
        self.last_path_time = rospy.Time.now()

    def _blend_with_path_command(
        self, ftg_steer: float, ftg_speed: float, front_distance: float
    ) -> Tuple[float, float]:
        if not self.use_path_blending:
            return ftg_steer, ftg_speed
        if self.last_path_cmd is None:
            return ftg_steer, ftg_speed
        if (rospy.Time.now() - self.last_path_time).to_sec() > self.path_timeout:
            return ftg_steer, ftg_speed

        path_steer = float(self.last_path_cmd.drive.steering_angle)
        path_speed = float(self.last_path_cmd.drive.speed)
        path_steer = max(min(path_steer, self.max_steering_angle), -self.max_steering_angle)
        path_speed = max(min(path_speed, self.max_speed), 0.0)

        blend = self._compute_blend_weight(front_distance)
        if front_distance <= self.blend_full_distance:
            blend = 1.0
        elif front_distance <= self.blend_activate_distance:
            blend = max(blend, 0.7)
        if self.turn_blend_scale > 0.0 and self.max_steering_angle > 1e-6:
            turn_ratio = min(abs(path_steer) / self.max_steering_angle, 1.0)
            blend *= max(0.0, 1.0 - self.turn_blend_scale * turn_ratio)
        blended_steer = blend * ftg_steer + (1.0 - blend) * path_steer
        blended_speed = blend * ftg_speed + (1.0 - blend) * path_speed

        return blended_steer, blended_speed

    def _compute_blend_weight(self, front_distance: float) -> float:
        if front_distance >= self.blend_activate_distance:
            return 0.0
        if front_distance <= self.blend_full_distance:
            return 1.0
        span = self.blend_activate_distance - self.blend_full_distance
        if span <= 0.0:
            return 1.0
        ratio = (self.blend_activate_distance - front_distance) / span
        return np.clip(ratio, 0.0, 1.0)

    def publish_drive(self, speed: float, steering: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)


def main() -> None:
    rospy.init_node("ftg_node_sim")
    FollowTheGapNode()
    rospy.loginfo("Follow-The-Gap 시뮬레이터 노드가 시작되었습니다.")
    rospy.spin()


if __name__ == "__main__":
    main()
