#!/usr/bin/env python3
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import Pose2D
from tf2_geometry_msgs import do_transform_pose  # 필요 시

class BaseLinkPosePublisher:
    def __init__(self):
        rospy.init_node("base_link_pose_publisher")
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.pub = rospy.Publisher("/base_link_pose", Pose2D, queue_size=10)
        self.rate = rospy.Rate(20)

    def spin(self):
        while not rospy.is_shutdown():
            try:
                tf_stamped = self.buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.1))
                x = tf_stamped.transform.translation.x
                y = tf_stamped.transform.translation.y
                q = tf_stamped.transform.rotation
                yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

                msg = Pose2D()
                msg.x = x
                msg.y = y
                msg.theta = yaw
                self.pub.publish(msg)
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass
            self.rate.sleep()

if __name__ == "__main__":
    node = BaseLinkPosePublisher()
    node.spin()
