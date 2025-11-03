#!/usr/bin/env python3
"""
sim_3/scripts/minimal_sim.py

Ackermann 기반 2D 시뮬레이터의 토대가 되는 노드.
현재 단계에서 제공하는 기능은 다음과 같다.
  * /drive (AckermannDriveStamped) 명령 적분
  * /sim3/pose (PoseStamped), /sim3/odom (Odometry) 퍼블리시
  * odom -> base_link 동적 TF, (옵션) map -> odom 정적 TF 브로드캐스트
  * /map (OccupancyGrid) 구독 후 레이캐스팅을 통해 /sim3/scan (LaserScan) 발행
"""

import math
from dataclasses import dataclass

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
import tf.transformations as tft
import tf2_ros
import numpy as np


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
    publish_map_tf: bool
    laser_offset_x: float
    laser_offset_y: float
    laser_offset_yaw: float
    map_topic: str
    max_ray_length: float


class MinimalSimulator:
    def __init__(self) -> None:
        rospy.init_node("sim3_minimal")
        self.params = self._load_params()
        self.params.max_ray_length = max(self.params.max_ray_length, self.params.scan_max_range)

        self.x = self.params.initial_x
        self.y = self.params.initial_y
        self.yaw = self.params.initial_yaw
        self.speed = 0.0
        self.steer = 0.0
        self.last_time = rospy.Time.now()

        self.pose_pub = rospy.Publisher(self.params.pose_topic, PoseStamped, queue_size=10)
        self.odom_pub = rospy.Publisher(self.params.odom_topic, Odometry, queue_size=10)
        self.gt_pose_pub = None
        if self.params.gt_pose_topic:
            self.gt_pose_pub = rospy.Publisher(self.params.gt_pose_topic, PoseStamped, queue_size=10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._publish_static_tf()

        rospy.Subscriber("/drive", AckermannDriveStamped, self._drive_callback)
        self.scan_pub = rospy.Publisher(self.params.scan_topic, LaserScan, queue_size=10)
        rospy.Subscriber(self.params.map_topic, OccupancyGrid, self._map_callback, queue_size=1)

        self.scan_angles = np.linspace(
            self.params.scan_min_angle,
            self.params.scan_max_angle,
            self.params.scan_count,
            dtype=np.float32,
        )
        self.map_info = None
        self.map_grid = None

        period = 1.0 / self.params.update_rate
        rospy.Timer(rospy.Duration(period), self._on_timer)

        rospy.loginfo(
            "[sim_3] Minimal simulator ready (x=%.2f, y=%.2f, yaw=%.2f°)",
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

        yaw_rate = 0.0
        if abs(self.params.wheelbase) > 1e-6:
            yaw_rate = self.speed * math.tan(self.steer) / self.params.wheelbase

        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.yaw = self._normalize_angle(self.yaw + yaw_rate * dt)

        quat = Quaternion(*tft.quaternion_from_euler(0.0, 0.0, self.yaw))
        self._publish_pose(now, quat)
        self._publish_odom(now, quat, yaw_rate)
        self._publish_tf(now, quat)
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

    def _publish_odom(self, stamp: rospy.Time, quat: Quaternion, yaw_rate: float) -> None:
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

    def _publish_tf(self, stamp: rospy.Time, quat: Quaternion) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.params.odom_frame
        tf_msg.child_frame_id = self.params.base_frame
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.rotation = quat
        self.tf_broadcaster.sendTransform(tf_msg)

    def _publish_scan(self, stamp: rospy.Time) -> None:
        if self.map_info is None or self.map_grid is None:
            return

        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.params.scan_frame
        scan.angle_min = self.params.scan_min_angle
        scan.angle_max = self.params.scan_max_angle
        scan.angle_increment = (
            (self.params.scan_max_angle - self.params.scan_min_angle)
            / max(1, self.params.scan_count - 1)
        )
        scan.range_min = 0.05
        scan.range_max = self.params.scan_max_range

        ranges = np.full(self.params.scan_count, self.params.scan_max_range, dtype=np.float32)

        res = self.map_info.resolution
        origin = self.map_info.origin
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        # base_link → laser 프레임 변환 (2D)
        laser_x = (
            self.x
            + cos_yaw * self.params.laser_offset_x
            - sin_yaw * self.params.laser_offset_y
        )
        laser_y = (
            self.y
            + sin_yaw * self.params.laser_offset_x
            + cos_yaw * self.params.laser_offset_y
        )
        laser_yaw = self.yaw + self.params.laser_offset_yaw

        ray_max = max(self.params.max_ray_length, self.params.scan_max_range)
        steps = max(2, int(ray_max / max(res, 1e-3)))

        for i, relative_angle in enumerate(self.scan_angles):
            beam_angle = laser_yaw + relative_angle
            cos_beam = math.cos(beam_angle)
            sin_beam = math.sin(beam_angle)

            hit_range = ray_max
            for dist in np.linspace(0.0, ray_max, steps):
                px = laser_x + dist * cos_beam
                py = laser_y + dist * sin_beam
                gx = int((px - origin.position.x) / res)
                gy = int((py - origin.position.y) / res)
                if (
                    gx < 0
                    or gy < 0
                    or gx >= self.map_info.width
                    or gy >= self.map_info.height
                ):
                    hit_range = dist
                    break
                if self.map_grid[gy, gx] > 50:
                    hit_range = dist
                    break
            ranges[i] = max(scan.range_min, min(self.params.scan_max_range, hit_range))

        if self.params.scan_noise > 0.0:
            ranges += np.random.normal(0.0, self.params.scan_noise, size=ranges.shape)
            ranges = np.clip(ranges, scan.range_min, scan.range_max)

        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)

    def _publish_static_tf(self) -> None:
        now = rospy.Time.now()
        if self.params.publish_map_tf:
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
        laser_tf.transform.translation.x = self.params.laser_offset_x
        laser_tf.transform.translation.y = self.params.laser_offset_y
        quat = tft.quaternion_from_euler(0.0, 0.0, self.params.laser_offset_yaw)
        laser_tf.transform.rotation.x = quat[0]
        laser_tf.transform.rotation.y = quat[1]
        laser_tf.transform.rotation.z = quat[2]
        laser_tf.transform.rotation.w = quat[3]
        self.static_broadcaster.sendTransform(laser_tf)

    def _map_callback(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.map_info = msg.info
        self.map_grid = data
        rospy.loginfo_once("[sim_3] Occupancy grid 수신 완료 (w=%d, h=%d)", msg.info.width, msg.info.height)

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
            pose_topic=rospy.get_param("~pose_topic", "/sim3/pose"),
            gt_pose_topic=rospy.get_param("~gt_pose_topic", "/gt_pose"),
            odom_topic=rospy.get_param("~odom_topic", "/sim3/odom"),
            scan_topic=rospy.get_param("~scan_topic", "/scan"),
            scan_frame=rospy.get_param("~scan_frame", "laser"),
            scan_min_angle=rospy.get_param("~scan_min_angle", math.radians(-135.0)),
            scan_max_angle=rospy.get_param("~scan_max_angle", math.radians(135.0)),
            scan_count=rospy.get_param("~scan_count", 1080),
            scan_max_range=rospy.get_param("~scan_max_range", 25.0),
            scan_noise=rospy.get_param("~scan_noise", 0.02),
            publish_map_tf=rospy.get_param("~publish_map_tf", True),
            laser_offset_x=rospy.get_param("~laser_offset_x", 0.27),
            laser_offset_y=rospy.get_param("~laser_offset_y", 0.0),
            laser_offset_yaw=rospy.get_param("~laser_offset_yaw", 0.0),
            map_topic=rospy.get_param("~map_topic", "/map"),
            max_ray_length=rospy.get_param("~max_ray_length", 30.0),
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


def main() -> None:
    try:
        MinimalSimulator().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
