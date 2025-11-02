#!/usr/bin/env python3
import csv
from pathlib import Path

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


def load_points(path: Path):
    points = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except ValueError:
                continue
            points.append((x, y))
    return points


def main() -> None:
    rospy.init_node("centerline_marker")

    default_path = Path("src/kora_k3/src/path_planning/outputs/map_center_1.csv")
    csv_path = Path(rospy.get_param("~centerline_file", str(default_path))).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()

    points = load_points(csv_path)
    if not points:
        rospy.logwarn(f"[centerline_marker] no points loaded from {csv_path}")

    pub = rospy.Publisher("~marker", Marker, queue_size=1)
    topic_name = rospy.resolve_name("~marker")
    rospy.loginfo(f"[centerline_marker] publishing {len(points)} points on {topic_name}")

    marker = Marker()
    marker.header.frame_id = rospy.get_param("~frame_id", "map")
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = rospy.get_param("~line_width", 0.2)
    marker.color.r = rospy.get_param("~color_r", 0.0)
    marker.color.g = rospy.get_param("~color_g", 1.0)
    marker.color.b = rospy.get_param("~color_b", 1.0)
    marker.color.a = rospy.get_param("~color_a", 1.0)

    z_height = rospy.get_param("~z_height", 0.02)
    for x, y in points:
        marker.points.append(Point(x=x, y=y, z=z_height))

    rate = rospy.Rate(rospy.get_param("~publish_rate", 2.0))
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        pub.publish(marker)
        rate.sleep()


if __name__ == "__main__":
    main()
