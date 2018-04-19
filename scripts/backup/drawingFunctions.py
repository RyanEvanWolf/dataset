#!/usr/bin/env python
from front_end.msg import StereoFrame,StereoMatch, Feature
from sensor_msgs.msg import Image,RegionOfInterest
import cv2
from cv_bridge import CvBridge
import rospy
import numpy as np
import copy
from threading import Thread,Lock

from bumblebee.srv import getOffset,getOffsetRequest,getOffsetResponse

class stereoDrawing():
    def __init__(self,leftROItopic,rightROItopic,leftRecttopic,rightRecttopic):
        self.cvb = CvBridge()
        self.lROISub = rospy.Subscriber(leftROItopic, Image, self.updatelROI)
        self.rROISub = rospy.Subscriber(rightROItopic, Image, self.updaterROI)
        self.lRectSub = rospy.Subscriber(leftRecttopic, Image, self.updatelRect)
        self.rRectSub = rospy.Subscriber(rightRecttopic, Image, self.updaterRect)
        self.MATlroi=np.zeros(100, dtype=np.uint8)
        self.MATrroi = np.zeros(100, dtype=np.uint8)
        self.MATlrect = np.zeros(100, dtype=np.uint8)
        self.MATrrect = np.zeros(100, dtype=np.uint8)
        self.getinfoServ=rospy.ServiceProxy("bumblebee_configuration/getOffset",getOffset)

        req=getOffsetRequest()
        self.info=self.getinfoServ(req)
    def updatelROI(self,message):
        self.MATlroi=copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updaterROI(self,message):
        self.MATrroi = copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updatelRect(self,message):
        self.MATlrect=copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updaterRect(self,message):
        self.MATrrect=copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updatelinfo(self,message):
        self.LroiInfo=copy.deepcopy(message)
    def updaterinfo(self,message):
        self.RroiInfo=copy.deepcopy(message)
    def drawMatches(self,matches):
        combinedImage=np.concatenate((self.MATlrect,self.MATrrect),axis=1)
        combinedImage=cv2.cvtColor(combinedImage,cv2.COLOR_GRAY2RGB)
        for index in matches:
            #get offset
            leftPoint_x=index.leftFeature.imageCoord.x + self.info.left.x_offset
            leftPoint_y=index.leftFeature.imageCoord.y+self.info.left.y_offset

            rightPoint_x=index.rightFeature.imageCoord.x+self.info.right.x_offset+1024
            rightPoint_y=index.rightFeature.imageCoord.y+self.info.right.y_offset
            ##draw onto image
            cv2.circle(combinedImage, (int(rightPoint_x), int(rightPoint_y)), 2, (0, 255, 0), -1)
            cv2.circle(combinedImage, (int(leftPoint_x), int(leftPoint_y)), 2, (0, 255, 255), -1)
            cv2.line(combinedImage, (int(leftPoint_x), int(leftPoint_y)), (int(rightPoint_x), int(rightPoint_y)), (0, 255, 255), 1)
        return copy.deepcopy(combinedImage)



class ImageThread():
        def __init__(self,displayName):
            self.imgMutex=Lock()
            self.imgName=displayName
            self.img=np.zeros(100, dtype=np.uint8)
            self.displayThread=Thread(target=self.updateLoop)
            self.displayThread.daemon=True
            self.visible=True
            self.displayThread.start()
        def updateLoop(self):
            while(True):
                if(self.visible):
                    cv2.namedWindow(self.imgName,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    self.imgMutex.acquire()
                    cv2.imshow(self.imgName,self.img)
                    self.imgMutex.release()
                    cv2.waitKey(20)
                else:
                    cv2.destroyWindow(self.imgName)
        def updateImage(self,newImage):
            self.imgMutex.acquire()
            self.img=copy.deepcopy(newImage)
            self.imgMutex.release()

class tracking():
    def __init__(self):
        self.cvb = CvBridge()
        self.lROISub = rospy.Subscriber("bumblebee/leftROI", Image, self.update)
        self.imgList=[]
        self.outImg=np.zeros(100, dtype=np.uint8)
        self.outMutex=Lock()
        self.maskSub=rospy.Subscriber("bumblebee/leftROI", Image, self.updateMask)
        ##mask to subscribe to
        ##stereoMatches to subsribe to 

        ##outImage
        ##window
    def update(self,message):
        self.img.append(copy.deepcopy(self.cvb.imgmsg_to_cv2(message)))
        if(len(self.imgList)>2):
            self.imgList.pop(0)
    def updateMask(self,message):
        print("abc")

