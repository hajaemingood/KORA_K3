#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


def load_waypoints(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        rospy.logerr(f"[waypoint_publisher] waypoint file not found: {csv_path}")
        return np.empty((0, 2), dtype=np.float32)

    data: List[Tuple[float, float]] = []
    with csv_path.open("r") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        has_header = False
        if header:
            try:
                float(header[0])
            except (ValueError, TypeError):
                has_header = True
        if not has_header:
            fh.seek(0)
            reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except ValueError:
                continue
            data.append((x, y))
    if not data:
        rospy.logwarn("[waypoint_publisher] waypoint file is empty.")
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def quaternion_to_yaw(q) -> float:
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


class WaypointPublisher:
    def __init__(self) -> None:
        rospy.init_node("waypoint_publisher")

        script_dir = Path(__file__).resolve().parent
        default_file = script_dir.parent / "src/path_planning/outputs/waypoints.csv"
        waypoint_param = rospy.get_param("~waypoint_file", str(default_file))
        csv_candidate = Path(str(waypoint_param)).expanduser()
        if csv_candidate.is_absolute():
            self.csv_path = csv_candidate
        else:
            if csv_candidate.exists():
                self.csv_path = csv_candidate.resolve()
            else:
                cwd_candidate = (Path.cwd() / csv_candidate).resolve()
                if cwd_candidate.exists():
                    self.csv_path = cwd_candidate
                else:
                    self.csv_path = (script_dir / csv_candidate).resolve()

        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.8)
        self.loop_path = rospy.get_param("~loop_path", True)
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)

        self.waypoints = load_waypoints(self.csv_path)
        if self.waypoints.size == 0:
            rospy.logfatal("[waypoint_publisher] No waypoints loaded; shutting down.")
            rospy.signal_shutdown("no waypoints")
            return

        self.current_index = 0
        self.last_position = None
        self.latest_pose = None

        self.waypoint_topic = rospy.get_param("~waypoint_topic", "/ftg_waypoint")
        self.pub = rospy.Publisher(self.waypoint_topic, Point, queue_size=1)
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        period = 1.0 / self.publish_rate if self.publish_rate > 0 else 0.05
        if self.publish_rate <= 0:
            rospy.logwarn("[waypoint_publisher] publish_rate must be > 0. Using 20Hz fallback.")
        self.timer = rospy.Timer(rospy.Duration(period), self._on_timer)
        rospy.loginfo(f"[waypoint_publisher] Loaded {len(self.waypoints)} waypoints from {self.csv_path}")

    def odom_callback(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.latest_pose = (position.x, position.y, quaternion_to_yaw(orientation))

    def _on_timer(self, event) -> None:
        if self.latest_pose is None or self.waypoints.size == 0:
            return
        car_x, car_y, car_yaw = self.latest_pose
        lookahead_point, self.current_index = self._find_lookahead(car_x, car_y)
        if lookahead_point is None:
            rospy.logwarn_throttle(5.0, "[waypoint_publisher] Failed to find lookahead point.")
            return

        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y

        cos_yaw = math.cos(car_yaw)
        sin_yaw = math.sin(car_yaw)
        x_rel = cos_yaw * dx + sin_yaw * dy
        y_rel = -sin_yaw * dx + cos_yaw * dy

        msg = Point()
        msg.x = x_rel
        msg.y = y_rel
        msg.z = 0.0
        self.pub.publish(msg)

    def _find_lookahead(self, car_x: float, car_y: float) -> Tuple[np.ndarray, int]:
        total_points = len(self.waypoints)
        if self.last_position is None:
            dists = np.linalg.norm(self.waypoints - np.array([car_x, car_y]), axis=1)
            nearest_index = int(np.argmin(dists))
        else:
            start = self.current_index
            end = start + min(200, len(self.waypoints))
            indices = np.arange(start, end) % total_points
            subset = self.waypoints[indices]
            dists = np.linalg.norm(subset - np.array([car_x, car_y]), axis=1)
            nearest_index = indices[int(np.argmin(dists))]

        if not self.loop_path and nearest_index >= total_points - 1:
            self.last_position = (car_x, car_y)
            return self.waypoints[-1], total_points - 1

        cumulative = 0.0
        index = nearest_index
        while cumulative < self.lookahead_distance:
            next_index = index + 1
            if not self.loop_path and next_index >= total_points:
                break
            next_index %= total_points
            segment = np.linalg.norm(self.waypoints[next_index] - self.waypoints[index])
            cumulative += segment
            index = next_index
            if not self.loop_path and index >= total_points - 1:
                break

        self.last_position = (car_x, car_y)
        return self.waypoints[index], index


def main() -> None:
    WaypointPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
