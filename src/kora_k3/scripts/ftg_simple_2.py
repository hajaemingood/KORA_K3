#!/usr/bin/env python3
import math
from pathlib import Path
import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


def angle_saturate(value, limit):
    return float(np.clip(value, -limit, limit))


class SimpleFTGNode:

    def __init__(self) -> None:
        rospy.init_node("ftg_simple")

        # 파라미터
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.drive_topic = rospy.get_param("~drive_topic", "/drive")
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.34)
        self.max_speed = rospy.get_param("~max_speed", 3.0)
        self.min_speed = rospy.get_param("~min_speed", 1)
        self.smoothing_window = rospy.get_param("~smoothing_window", 5)
        self.forward_angle = np.deg2rad(rospy.get_param("~forward_angle_deg", 90))
        self.bubble_size = rospy.get_param("~bubble_size", 10)
        self.valid_threshold = rospy.get_param("~valid_distance", 0.8)
        self.path_weight = rospy.get_param("~path_weight", 0.6)
        self.default_heading = angle_saturate(
            rospy.get_param("~default_heading", 0.0),
            self.max_steering_angle
        )
        self.waypoint_timeout = rospy.get_param("~waypoint_timeout", 1.0)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 1.0)
        self.loop_centerline = rospy.get_param("~loop_centerline", True)
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.blend_distance = rospy.get_param("~blend_distance", 1.2)

        # 상태 변수
        self.waypoint_heading = self.default_heading
        self.last_waypoint_time = rospy.Time(0)
        self.have_waypoint = False
        self.angle_increment = None
        self.angle_min = None
        map_file = rospy.get_param(
            "~centerline_file",
            "src/kora_k3/src/path_planning/outputs/map_center_1.csv",
        )
        self.centerline = self._load_centerline(map_file)
        self.center_index = 0
        self.last_pose_xy = None

        # ROS 통신
        rospy.Subscriber("~waypoint", Point, self.waypoint_callback, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

    # Waypoint 콜백
    def waypoint_callback(self, msg: Point) -> None:
        """외부 waypoint → heading 계산"""
        heading = self.default_heading
        if not math.isclose(msg.x, 0.0, abs_tol=1e-6) or not math.isclose(msg.y, 0.0, abs_tol=1e-6):
            heading = math.atan2(msg.y, msg.x)
        elif not math.isclose(msg.z, 0.0, abs_tol=1e-6):
            heading = msg.z

        self.waypoint_heading = angle_saturate(heading, self.max_steering_angle)
        self.last_waypoint_time = rospy.Time.now()
        self.have_waypoint = True

    # LiDAR 콜백
    def scan_callback(self, scan: LaserScan) -> None:
        """LiDAR 스캔 데이터 처리 및 조향/속도 결정"""
        if np.isnan(scan.angle_increment) or scan.angle_increment == 0.0:
            rospy.logwarn_throttle(5.0, "invalid scan")
            return

        # 초기 각도 파라미터 설정
        if self.angle_increment is None:
            self.angle_increment = scan.angle_increment
            self.angle_min = scan.angle_min

        # 거리 및 각도 배열
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = self.angle_min + np.arange(len(ranges)) * self.angle_increment

        # 전방 영역만 선택
        mask = np.abs(angles) <= self.forward_angle
        ranges = ranges[mask]
        angles = angles[mask]

        if ranges.size == 0:
            self.publish_drive(self.min_speed, self._current_waypoint_heading())
            return

        # smoothing
        kernel = np.ones(self.smoothing_window) / float(self.smoothing_window)
        ranges = np.convolve(ranges, kernel, mode="same")

        # 가장 가까운 장애물 bubble masking
        closest_idx = np.argmin(ranges)
        if ranges[closest_idx] <= self.valid_threshold:
            start_bubble = max(0, closest_idx - self.bubble_size)
            end_bubble = min(len(ranges), closest_idx + self.bubble_size)
            ranges[start_bubble:end_bubble] = 0.0

        # 유효 구간 마스크
        valid = ranges > self.valid_threshold
        waypoint_heading = self._current_waypoint_heading()

        if not np.any(valid):
            self.publish_drive(self.min_speed, waypoint_heading)
            return

        front_distance = float(np.min(ranges[valid])) if np.any(valid) else self.valid_threshold

        # gap 검출 (연속된 True 구간)
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

        # 가장 넓은 gap 선택
        best_gap = max(gaps, key=lambda s: s[1] - s[0])
        gap_start, gap_end = best_gap
        mid_index = int((gap_start + gap_end) * 0.5)
        gap_heading = angles[mid_index]

        # gap-heading과 waypoint-heading 블렌딩
        blend_weight = self.path_weight if self._has_recent_waypoint() else 0.0
        if front_distance < self.blend_distance:
            blend_weight = 0.0
        blended_heading = (1.0 - blend_weight) * gap_heading + blend_weight * waypoint_heading

        # 조향 제한
        steering = angle_saturate(blended_heading, self.max_steering_angle)

        # 조향 크기에 따른 속도 보정
        ratio = min(abs(steering) / self.max_steering_angle, 1.0)
        speed = self.max_speed * (1.0 - ratio)
        speed = max(speed, self.min_speed)

        self.publish_drive(speed, steering)

    # -------------------------------------------
    # 명령 퍼블리시
    # -------------------------------------------
    def publish_drive(self, speed: float, steering: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)

    # -------------------------------------------
    # 헬퍼 함수
    # -------------------------------------------
    def _has_recent_waypoint(self) -> bool:
        if not self.have_waypoint:
            return False
        return (rospy.Time.now() - self.last_waypoint_time).to_sec() <= self.waypoint_timeout

    def _current_waypoint_heading(self) -> float:
        if self._has_recent_waypoint():
            return self.waypoint_heading
        return self.default_heading

    def odom_callback(self, msg: Odometry) -> None:
        if self.centerline.size == 0:
            return
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ])[2]
        lookahead_point, self.center_index = self._find_lookahead_point(position.x, position.y)
        if lookahead_point is None:
            return
        dx = lookahead_point[0] - position.x
        dy = lookahead_point[1] - position.y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_rel = cos_yaw * dx + sin_yaw * dy
        y_rel = -sin_yaw * dx + cos_yaw * dy
        heading = math.atan2(y_rel, x_rel)
        self.waypoint_heading = angle_saturate(heading, self.max_steering_angle)
        self.last_waypoint_time = rospy.Time.now()
        self.have_waypoint = True
        self.last_pose_xy = (position.x, position.y)

    def _find_lookahead_point(self, x: float, y: float):
        points = self.centerline
        if points.size == 0:
            return (None, self.center_index)
        total = len(points)
        if self.last_pose_xy is None:
            dists = np.linalg.norm(points - np.array([x, y]), axis=1)
            nearest_index = int(np.argmin(dists))
        else:
            start = self.center_index
            span = min(total, 200)
            indices = (np.arange(start, start + span)) % total
            subset = points[indices]
            dists = np.linalg.norm(subset - np.array([x, y]), axis=1)
            nearest_index = indices[int(np.argmin(dists))]
        cumulative = 0.0
        index = nearest_index
        while cumulative < self.lookahead_distance:
            next_index = index + 1
            if not self.loop_centerline and next_index >= total:
                break
            next_index %= total
            segment = np.linalg.norm(points[next_index] - points[index])
            cumulative += segment
            index = next_index
            if not self.loop_centerline and index >= total - 1:
                break
        return points[index], index

    def _load_centerline(self, file_path: str) -> np.ndarray:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            rospy.logwarn(f"[ftg_simple_2] centerline file not found: {path}")
            return np.empty((0, 2), dtype=np.float32)
        try:
            data = np.genfromtxt(path, delimiter=",", skip_header=1)
        except Exception as exc:
            rospy.logerr(f"[ftg_simple_2] failed to load centerline: {exc}")
            return np.empty((0, 2), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            rospy.logwarn("[ftg_simple_2] centerline file has insufficient columns")
            return np.empty((0, 2), dtype=np.float32)
        rospy.loginfo(f"[ftg_simple_2] Loaded {data.shape[0]} centerline points from {path}")
        return data[:, :2].astype(np.float32)


def main() -> None:
    SimpleFTGNode()
    rospy.spin()


if __name__ == "__main__":
    main()
