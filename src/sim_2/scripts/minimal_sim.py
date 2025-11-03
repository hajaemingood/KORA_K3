#!/usr/bin/env python3
"""
sim_2/scripts/minimal_sim.py

가장 단순한 2D Ackermann 시뮬레이터.
- /drive (AckermannDriveStamped) 명령을 받아 차량 상태 적분
- /sim2/pose (PoseStamped), /sim2/odom (Odometry) 퍼블리시
- odom -> base_link TF 브로드캐스트, map -> odom 정적 TF(ident.)

이 단계에서는 맵/센서 모델 없이 차량 운동만 확인한다.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
import tf.transformations as tft
import tf2_ros
from sensor_msgs.msg import LaserScan


@dataclass
class SimParams:
    initial_x: float
    initial_y: float
    initial_yaw: float
    wheelbase: float
    update_rate: float
    map_frame: str
    odom_frame: str
    base_frame: str
    pose_topic: str
    gt_pose_topic: str
    odom_topic: str
    scan_topic: str
    scan_frame: str
    scan_min_angle: float
    scan_max_angle: float
    scan_count: int
    scan_max_range: float
    scan_noise: float
    grid_resolution: float
    grid_width: int
    grid_height: int
    grid_origin_x: float
    grid_origin_y: float
    publish_dummy_map: bool


class MinimalSimulator:
    def __init__(self) -> None:
        rospy.init_node("sim2_minimal")
        self.params = self._load_params()

        # 상태 변수
        self.x = self.params.initial_x
        self.y = self.params.initial_y
        self.yaw = self.params.initial_yaw
        self.speed = 0.0
        self.steer = 0.0
        self.last_time = rospy.Time.now()
        self.scan_angles = np.linspace(
            self.params.scan_min_angle,
            self.params.scan_max_angle,
            self.params.scan_count,
            dtype=np.float32,
        )
        self.gt_pose_pub = None
        if self.params.gt_pose_topic:
            self.gt_pose_pub = rospy.Publisher(
                self.params.gt_pose_topic, PoseStamped, queue_size=10
            )
        self.map_grid = self._create_dummy_grid() if self.params.publish_dummy_map else None

        # 퍼블리셔
        self.pose_pub = rospy.Publisher(
            self.params.pose_topic, PoseStamped, queue_size=10
        )
        self.odom_pub = rospy.Publisher(
            self.params.odom_topic, Odometry, queue_size=10
        )
        self.scan_pub = rospy.Publisher(
            self.params.scan_topic, LaserScan, queue_size=5
        )
        self.map_pub = None
        if self.map_grid is not None:
            self.map_pub = rospy.Publisher("sim2/map", OccupancyGrid, queue_size=1, latch=True)

        # TF 브로드캐스터
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._publish_static_tf()

        # 명령 구독
        rospy.Subscriber("/drive", AckermannDriveStamped, self._drive_callback)

        # 주기적 업데이트
        period = 1.0 / self.params.update_rate
        rospy.Timer(rospy.Duration(period), self._on_timer)
        if self.map_grid is not None and self.map_pub is not None:
            self.map_pub.publish(self.map_grid)

        rospy.loginfo(
            "[sim_2] Minimal simulator ready (x=%.2f, y=%.2f, yaw=%.2f°)",
            self.x,
            self.y,
            math.degrees(self.yaw),
        )

    def spin(self) -> None:
        rospy.spin()

    def _drive_callback(self, msg: AckermannDriveStamped) -> None:
        self.speed = float(msg.drive.speed)
        self.steer = float(msg.drive.steering_angle)

    def _on_timer(self, _event) -> None:
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        if dt <= 0.0:
            return
        self.last_time = now

        # 단순 Ackermann 모델 적분
        yaw_rate = 0.0
        if abs(self.params.wheelbase) > 1e-6:
            yaw_rate = self.speed * math.tan(self.steer) / self.params.wheelbase

        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.yaw = self._normalize_angle(self.yaw + yaw_rate * dt)

        quat = Quaternion(*tft.quaternion_from_euler(0.0, 0.0, self.yaw))
        self._publish_pose(now, quat)
        self._publish_odom(now, quat, yaw_rate)
        self._publish_tf(now, quat, yaw_rate)
        self._publish_scan(now)

    def _publish_pose(self, stamp: rospy.Time, quat: Quaternion) -> None:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.params.map_frame
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.orientation = quat
        self.pose_pub.publish(pose)
        if self.gt_pose_pub is not None:
            self.gt_pose_pub.publish(pose)

    def _publish_odom(
        self, stamp: rospy.Time, quat: Quaternion, yaw_rate: float
    ) -> None:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.params.odom_frame
        odom.child_frame_id = self.params.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = quat
        odom.twist.twist.linear.x = self.speed
        odom.twist.twist.angular.z = yaw_rate
        self.odom_pub.publish(odom)

    def _publish_tf(
        self, stamp: rospy.Time, quat: Quaternion, yaw_rate: float
    ) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.params.odom_frame
        tf_msg.child_frame_id = self.params.base_frame
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.rotation = quat
        self.tf_broadcaster.sendTransform(tf_msg)

        map_tf = TransformStamped()
        map_tf.header.stamp = stamp
        map_tf.header.frame_id = self.params.map_frame
        map_tf.child_frame_id = self.params.odom_frame
        map_tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(map_tf)

    def _publish_static_tf(self) -> None:
        now = rospy.Time.now()
        static_tf = TransformStamped()
        static_tf.header.stamp = now
        static_tf.header.frame_id = self.params.map_frame
        static_tf.child_frame_id = self.params.odom_frame
        static_tf.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(static_tf)

        laser_tf = TransformStamped()
        laser_tf.header.stamp = now
        laser_tf.header.frame_id = self.params.base_frame
        laser_tf.child_frame_id = self.params.scan_frame
        laser_tf.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(laser_tf)

    def _load_params(self) -> SimParams:
        return SimParams(
            initial_x=rospy.get_param("~initial_x", 0.0),
            initial_y=rospy.get_param("~initial_y", 0.0),
            initial_yaw=rospy.get_param("~initial_yaw", 0.0),
            wheelbase=rospy.get_param("~wheelbase", 0.33),
            update_rate=rospy.get_param("~update_rate", 50.0),
            map_frame=rospy.get_param("~map_frame", "map"),
            odom_frame=rospy.get_param("~odom_frame", "odom"),
            base_frame=rospy.get_param("~base_frame", "base_link"),
            pose_topic=rospy.get_param("~pose_topic", "/sim2/pose"),
            gt_pose_topic=rospy.get_param("~gt_pose_topic", "/gt_pose"),
            odom_topic=rospy.get_param("~odom_topic", "/sim2/odom"),
            scan_topic=rospy.get_param("~scan_topic", "/sim2/scan"),
            scan_frame=rospy.get_param("~scan_frame", "laser"),
            scan_min_angle=rospy.get_param("~scan_min_angle", -math.pi),
            scan_max_angle=rospy.get_param("~scan_max_angle", math.pi),
            scan_count=rospy.get_param("~scan_count", 360),
            scan_max_range=rospy.get_param("~scan_max_range", 20.0),
            scan_noise=rospy.get_param("~scan_noise", 0.01),
            grid_resolution=rospy.get_param("~grid_resolution", 0.2),
            grid_width=rospy.get_param("~grid_width", 200),
            grid_height=rospy.get_param("~grid_height", 200),
            grid_origin_x=rospy.get_param("~grid_origin_x", -20.0),
            grid_origin_y=rospy.get_param("~grid_origin_y", -20.0),
            publish_dummy_map=rospy.get_param("~publish_dummy_map", True),
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _create_dummy_grid(self) -> Optional[OccupancyGrid]:
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.params.map_frame
        grid.info.resolution = self.params.grid_resolution
        grid.info.width = self.params.grid_width
        grid.info.height = self.params.grid_height
        grid.info.origin.position.x = self.params.grid_origin_x
        grid.info.origin.position.y = self.params.grid_origin_y
        grid.info.origin.orientation.w = 1.0

        data = np.full(
            (self.params.grid_height, self.params.grid_width), 0, dtype=np.int8
        )
        cx = self.params.grid_width // 2
        cy = self.params.grid_height // 2
        radius = int(5.0 / self.params.grid_resolution)
        for angle in np.linspace(0.0, 2 * math.pi, 360):
            for r in range(radius, radius + 5):
                ix = int(cx + r * math.cos(angle))
                iy = int(cy + r * math.sin(angle))
                if 0 <= ix < self.params.grid_width and 0 <= iy < self.params.grid_height:
                    data[iy, ix] = 100
        grid.data = data.flatten().tolist()
        return grid

    def _publish_scan(self, stamp: rospy.Time) -> None:
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.params.scan_frame
        scan.angle_min = self.params.scan_min_angle
        scan.angle_max = self.params.scan_max_angle
        scan.angle_increment = (
            (self.params.scan_max_angle - self.params.scan_min_angle)
            / (self.params.scan_count - 1)
        )
        scan.range_min = 0.05
        scan.range_max = self.params.scan_max_range

        ranges = np.full(self.params.scan_count, self.params.scan_max_range, dtype=np.float32)
        res = self.params.grid_resolution
        ox = self.params.grid_origin_x
        oy = self.params.grid_origin_y
        grid = (
            np.array(self.map_grid.data, dtype=np.int8).reshape(
                self.params.grid_height, self.params.grid_width
            )
            if self.map_grid is not None
            else None
        )
        robot_x = self.x
        robot_y = self.y
        for i, angle in enumerate(self.scan_angles):
            beam_angle = self.yaw + angle
            hit = self.params.scan_max_range
            for dist in np.linspace(0.0, self.params.scan_max_range, 200):
                px = robot_x + dist * math.cos(beam_angle)
                py = robot_y + dist * math.sin(beam_angle)
                gx = int((px - ox) / res)
                gy = int((py - oy) / res)
                if (
                    gx < 0
                    or gx >= self.params.grid_width
                    or gy < 0
                    or gy >= self.params.grid_height
                ):
                    hit = dist
                    break
                if grid is not None and grid[gy, gx] > 50:
                    hit = dist
                    break
            ranges[i] = max(scan.range_min, hit + np.random.normal(0.0, self.params.scan_noise))

        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)


def main() -> None:
    try:
        MinimalSimulator().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
