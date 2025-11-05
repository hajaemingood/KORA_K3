#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import rospy
from std_msgs.msg import Float64


def load_waypoints(csv_path: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with open(csv_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            points.append((x, y))
    return points


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class CommandSegment:
    duration: float
    motor: float
    servo: float


class TimedWaypointDriver:
    def __init__(self) -> None:
        rospy.init_node("timed_waypoint_driver")

        waypoint_file = rospy.get_param(
            "~waypoint_file",
            "/root/KORA_K3/src/kora_k3/src/path_planning/outputs/test.csv",
        )
        target_speed = rospy.get_param("~target_speed", 1.5)  # m/s
        motor_scale = rospy.get_param("~motor_scale", 3500.0)  # (m/s) → 모터 명령
        motor_limit = rospy.get_param("~motor_limit", 9000.0)
        max_steer_rad = rospy.get_param("~max_steer_rad", 0.5)
        servo_center = rospy.get_param("~servo_center", 0.5)
        servo_limit = rospy.get_param("~servo_limit", 0.5)

        waypoints = load_waypoints(waypoint_file)
        if len(waypoints) < 2:
            raise RuntimeError(f"웨이포인트가 부족합니다: {waypoint_file}")

        self.schedule = self.build_schedule(
            waypoints,
            target_speed=target_speed,
            motor_scale=motor_scale,
            motor_limit=motor_limit,
            max_steer_rad=max_steer_rad,
            servo_center=servo_center,
            servo_limit=servo_limit,
        )

        self.motor_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.servo_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        self.publish_rate = rospy.Rate(rospy.get_param("~publish_rate", 20.0))
        self.repeat = rospy.get_param("~repeat", False)

    def build_schedule(
        self,
        waypoints: Sequence[Tuple[float, float]],
        *,
        target_speed: float,
        motor_scale: float,
        motor_limit: float,
        max_steer_rad: float,
        servo_center: float,
        servo_limit: float,
    ) -> List[CommandSegment]:
        segments: List[CommandSegment] = []
        prev_heading = math.atan2(
            waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0]
        )

        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                continue

            heading = math.atan2(dy, dx)
            delta_heading = wrap_angle(heading - prev_heading)
            prev_heading = heading

            servo_offset = np.clip(delta_heading / max_steer_rad, -1.0, 1.0)
            servo_cmd = servo_center - servo_offset * servo_limit
            servo_cmd = float(np.clip(servo_cmd, servo_center - servo_limit, servo_center + servo_limit))

            duration = distance / max(target_speed, 1e-3)
            motor_cmd = float(np.clip(target_speed * motor_scale, 0.0, motor_limit))

            segments.append(CommandSegment(duration=duration, motor=motor_cmd, servo=servo_cmd))

        return segments

    def run(self) -> None:
        rospy.loginfo("[timed_waypoint_driver] %d segments scheduled", len(self.schedule))
        idx = 0
        while not rospy.is_shutdown():
            if idx >= len(self.schedule):
                if self.repeat:
                    idx = 0
                else:
                    break
            segment = self.schedule[idx]
            self._execute_segment(segment)
            idx += 1

        self._publish_command(0.0, 0.5)
        rospy.loginfo("[timed_waypoint_driver] finished. Vehicle stopped.")

    def _execute_segment(self, segment: CommandSegment) -> None:
        end_time = rospy.Time.now() + rospy.Duration(segment.duration)
        rospy.loginfo(
            "[timed_waypoint_driver] duration=%.2fs motor=%.1f servo=%.3f",
            segment.duration,
            segment.motor,
            segment.servo,
        )
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self._publish_command(segment.motor, segment.servo)
            self.publish_rate.sleep()

    def _publish_command(self, motor: float, servo: float) -> None:
        self.motor_pub.publish(Float64(motor))
        self.servo_pub.publish(Float64(servo))


if __name__ == "__main__":
    driver = TimedWaypointDriver()
    driver.run()
