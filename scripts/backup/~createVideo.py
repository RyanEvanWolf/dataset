#!/usr/bin/env python
from sensor_msgs.msg import Image
import sys
import rospy
import cv2
from cv_bridge import CvBridge
import copy
import numpy as np
from front_end.msg import StereoFrame,FrameTracks,Window
from std_msgs.msg import Bool
import time
import imageio

defaultOutDirectory="/media/ryan/Markov/playback"


width=int(sys.argv[2])
height=int(sys.argv[3])
global waitPub
waitPub=rospy.Publisher(str(sys.argv[1])+"/wait",Bool, queue_size=5)
print(sys.argv[1]+"/wait")
outVideoName=defaultOutDirectory+sys.argv[1][sys.argv[1].rfind('/'):]+".avi"
print("saving to "+outVideoName)
global vid
vid=cv2.VideoWriter(outVideoName,cv2.VideoWriter_fourcc(*'MPEG'),15,(width,height))
cvb = CvBridge()


def callback(message):
    global waitPub
    img=copy.deepcopy(cvb.imgmsg_to_cv2(message))
    if(len(img.shape)<3):
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    vid.write(img)
    message=Bool()
    waitPub.publish(message)



def updatel(self, message):
    self.limg = copy.deepcopy(self.cvb.imgmsg_to_cv2(message))

rospy.init_node("createVid")
imgSub=rospy.Subscriber(sys.argv[1],Image,callback)
wait=True
while(wait):
    wait = not rospy.get_param("/endSequence")
    time.sleep(0.1)
vid.release()
rospy.set_param("/endSequence",False)
print("finished recording " + sys.argv[1])