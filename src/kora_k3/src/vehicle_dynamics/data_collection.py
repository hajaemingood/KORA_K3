#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64

def main():
    rospy.init_node("data_collection_driver")
    motor_topic = rospy.get_param("~motor_topic", "/commands/motor/speed")
    steer_topic = rospy.get_param("~steer_topic", "/commands/servo/position")
    speed_cmd = rospy.get_param("~speed", 10000.0)
    steer_cmd = rospy.get_param("~steer", 0.0)
    duration = rospy.get_param("~duration", 30.0)
    publish_rate = rospy.get_param("~rate", 20.0)
    motor_pub = rospy.Publisher(motor_topic, Float64, queue_size=10)
    steer_pub = rospy.Publisher(steer_topic, Float64, queue_size=10)
    rospy.sleep(0.5)  # give publishers time to register
    start_time = rospy.get_time()
    
    while start_time == 0.0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
        start_time = rospy.get_time()
    rate = rospy.Rate(publish_rate)
    
    while not rospy.is_shutdown():
        elapsed = rospy.get_time() - start_time
        if elapsed >= duration:
            break
        motor_pub.publish(Float64(speed_cmd))
        steer_pub.publish(Float64(steer_cmd))
        rate.sleep()
    
    motor_pub.publish(Float64(0.0))
    steer_pub.publish(Float64(steer_cmd))

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass