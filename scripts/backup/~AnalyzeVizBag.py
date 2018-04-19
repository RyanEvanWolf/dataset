#!/usr/bin/env python

import rospy
import rosbag
from geometry_msgs.msg import PoseArray,Transform,TransformStamped,Point

from std_msgs.msg import String,Int32
from viso2_ros.msg import VisoInfo
import time
import tf
import math

from visualization_msgs.msg import Marker

rospy.init_node("analyze")
inBag=rosbag.Bag('/home/ryan/git/Output/visoBag.bag','r')




test=PoseArray()
test.header.frame_id="cv"

m=Marker()
mp=Marker()

pointLocations=[]

topCount=0
for topic, msg, t in inBag.read_messages():
    if(topic=="/lib_viso/pose"):
        test.poses.append(msg.pose)
        m.points.append(msg.pose.position)#.x,msg.pose.translation.y,msg.pose.translation.z))
        mp.points.append(msg.pose.position)
print(len(test.poses))
a=rospy.Publisher("set",PoseArray,queue_size=10)


br=tf.TransformBroadcaster()

toWorld=TransformStamped()
toWorld.header.frame_id="world"
toWorld.child_frame_id="cv"

toWorld.transform.translation.x=0
toWorld.transform.translation.y=0
toWorld.transform.translation.z=0

toWorld.transform.rotation= tf.transformations.quaternion_from_euler(math.radians(90),math.radians(90),0)


m.type=4
m.action=0
m.header.frame_id="world"


m.scale.x=0.01
m.scale.y=0.01
m.scale.z=0.01
m.color.g=1.0
m.pose.orientation.w=0
m.color.a=1

mp.type=6
mp.action=0
mp.header.frame_id="world"


mp.scale.x=0.01
mp.scale.y=0.01
mp.scale.z=0.01
mp.color.r=1.0
mp.pose.orientation.w=0
mp.color.a=1


p=rospy.Publisher("mark",Marker,queue_size=10)
pp=rospy.Publisher("marker2",Marker,queue_size=10)
p.publish(m)
pp.publish(mp)




while(not  rospy.is_shutdown()):
    time.sleep(0.5)
    m.header.stamp=rospy.Time.now()
    mp.header.stamp=rospy.Time.now()
    p.publish(m)
    pp.publish(mp)
#    toWorld.header.stamp=rospy.Time.now()
#    test.header.stamp=rospy.Time.now()
##    br.sendTransform((0,0,0),toWorld.transform.rotation,toWorld.header.stamp,toWorld.child_frame_id,toWorld.header.frame_id)
#    a.publish(test)


rospy.spin()