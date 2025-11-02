#!/usr/bin/env python3
import math
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point


def angle_saturate(value, limit):
    return float(np.clip(value, -limit, limit))


class SimpleFTGNode:
    def __init__(self) -> None:
        rospy.init_node("ftg_simple")

        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.drive_topic = rospy.get_param("~drive_topic", "/drive")
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.34)
        self.max_speed = rospy.get_param("~max_speed", 3.0)
        self.min_speed = rospy.get_param("~min_speed", 0.6)
        self.smoothing_window = rospy.get_param("~smoothing_window", 7)
        self.forward_angle = np.deg2rad(rospy.get_param("~forward_angle_deg", 70))
        self.bubble_size = rospy.get_param("~bubble_size", 8)
        self.valid_threshold = rospy.get_param("~valid_distance", 0.9)
        self.path_weight = rospy.get_param("~path_weight", 0.8)
        self.waypoint_topic = rospy.get_param("~waypoint_topic", "/ftg_waypoint")
        self.default_heading = angle_saturate(
            rospy.get_param("~default_heading", 0.0), self.max_steering_angle
        )
        self.waypoint_timeout = rospy.get_param("~waypoint_timeout", 1.0)
        self.center_clearance = rospy.get_param("~center_clearance", 0.25)

        self.waypoint_heading = self.default_heading
        self.last_waypoint_time = rospy.Time(0)
        self.have_waypoint = False
        rospy.Subscriber(self.waypoint_topic, Point, self.waypoint_callback, queue_size=1)

        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        self.angle_increment = None
        self.angle_min = None

    def waypoint_callback(self, msg: Point) -> None:
        heading = self.default_heading
        if not math.isclose(msg.x, 0.0, abs_tol=1e-6) or not math.isclose(msg.y, 0.0, abs_tol=1e-6):
            heading = math.atan2(msg.y, msg.x)
        elif not math.isclose(msg.z, 0.0, abs_tol=1e-6):
            heading = msg.z
        self.waypoint_heading = angle_saturate(heading, self.max_steering_angle)
        self.last_waypoint_time = rospy.Time.now()
        self.have_waypoint = True

    def scan_callback(self, scan: LaserScan) -> None:
        if np.isnan(scan.angle_increment) or scan.angle_increment == 0.0:
            rospy.logwarn_throttle(5.0, "invalid scan")
            return

        if self.angle_increment is None:
            self.angle_increment = scan.angle_increment
            self.angle_min = scan.angle_min

        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = self.angle_min + np.arange(len(ranges)) * self.angle_increment

        mask = np.abs(angles) <= self.forward_angle
        ranges = ranges[mask]
        angles = angles[mask]

        if ranges.size == 0:
            self.publish_drive(self.min_speed, self._current_waypoint_heading())
            return

        kernel = np.ones(self.smoothing_window) / float(self.smoothing_window)
        ranges = np.convolve(ranges, kernel, mode="same")

        closest_idx = np.argmin(ranges)
        if ranges[closest_idx] <= self.valid_threshold:
            start_bubble = max(0, closest_idx - self.bubble_size)
            end_bubble = min(len(ranges), closest_idx + self.bubble_size)
            ranges[start_bubble:end_bubble] = 0.0

        valid = ranges > self.valid_threshold
        waypoint_heading = self._current_waypoint_heading()
        if not np.any(valid):
            self.publish_drive(self.min_speed, waypoint_heading)
            return

        valid_indices = np.where(valid)[0]
        edges = np.diff(valid.astype(np.int8))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if valid[0]:
            starts.insert(0, 0)
        if valid[-1]:
            ends.append(len(valid) - 1)

        gaps = list(zip(starts, ends))
        if not gaps:
            self.publish_drive(self.min_speed, waypoint_heading)
            return

        best_gap = max(gaps, key=lambda s: s[1] - s[0])
        gap_start, gap_end = best_gap
        mid_index = int((gap_start + gap_end) * 0.5)
        gap_heading = angles[mid_index]

        if valid_indices.size > 0 and self._has_recent_waypoint():
            path_idx = self._select_path_index(valid_indices, angles, ranges, waypoint_heading)
            path_heading = angles[path_idx]
        else:
            path_heading = waypoint_heading

        blend_weight = self.path_weight if self._has_recent_waypoint() else 0.0
        blended_heading = (1.0 - blend_weight) * gap_heading + blend_weight * path_heading
        steering = angle_saturate(blended_heading, self.max_steering_angle)

        ratio = min(abs(steering) / self.max_steering_angle, 1.0)
        speed = self.max_speed * (1.0 - ratio)
        speed = max(speed, self.min_speed)

        self.publish_drive(speed, steering)

    def publish_drive(self, speed: float, steering: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)

    def _has_recent_waypoint(self) -> bool:
        if not self.have_waypoint:
            return False
        return (rospy.Time.now() - self.last_waypoint_time).to_sec() <= self.waypoint_timeout

    def _current_waypoint_heading(self) -> float:
        if self._has_recent_waypoint():
            return self.waypoint_heading
        return self.default_heading

    def _select_path_index(
        self,
        valid_indices: np.ndarray,
        angles: np.ndarray,
        ranges: np.ndarray,
        waypoint_heading: float,
    ) -> int:
        angle_diffs = np.abs(angles[valid_indices] - waypoint_heading)
        order = np.argsort(angle_diffs)
        for idx in order:
            candidate = valid_indices[idx]
            if ranges[candidate] >= (self.valid_threshold + self.center_clearance):
                return candidate
        return valid_indices[order[0]]


def main() -> None:
    SimpleFTGNode()
    rospy.spin()


if __name__ == "__main__":
    main()
