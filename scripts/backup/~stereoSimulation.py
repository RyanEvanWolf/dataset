#!/usr/bin/env python
#####ros related imports
import rospy
from getFeatures import getOrbParameters

from getFeatures import datasetFeatures,frameFeatures

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QImage
####custom packages imports
from bumbleDataSet import bumbleDataSetNode
from front_end.srv import setDetector,setDetectorResponse,setDetectorRequest
from front_end.msg import ORB,StereoFrame,FrameTracks,Window



from dataset.srv import resetDataset,resetDatasetResponse,resetDatasetRequest
from dataset.srv import getDatasetInfo,getDatasetInfoResponse,getDatasetInfoRequest
from dataset.srv import publishImage,publishImageResponse,publishImageRequest

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

##system imports
import sys
import copy
import cv2
import time
from threading import Thread, Lock
import numpy as np
import pickle


from std_msgs.msg import String

import signal


class FrameFeatures:
    def __init__(self):
        self.messages=[]
        self.responses=[]
        self.imageDir=[]

class DataSetFeatures:
    def __init__(self):
        self.frames = []

class StereoManager():
    def __init__(self,displayNames):
        ##wait for services
        rospy.wait_for_service("front_end/setDetector", 4)
        self.detSrv=rospy.ServiceProxy("front_end/setDetector",setDetector)
        try:
            rospy.wait_for_service("/dataset/publishImage", 4)
            print("Connected to /dataset/publishImage")
            self.publishSrv = rospy.ServiceProxy("/dataset/publishImage",publishImage)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            rospy.wait_for_service("/dataset/getDatasetInfo", 4)
            print("Connected to /dataset/getDatasetInfo")
            self.infoSrv = rospy.ServiceProxy("/dataset/getDatasetInfo",getDatasetInfo)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            rospy.wait_for_service("/dataset/resetDataset", 4)
            print("Connected to /dataset/resetDataset")
            self.resetSrv = rospy.ServiceProxy("/dataset/resetDataset", resetDataset)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        self.extractSettings=pickle.load(open("/media/ryan/Markov/DetectorOutput/D4/31_12_06-27-23-ORB/devFeatures.p", "rb"))
        #self.stereo = []
        self.stereoSub = rospy.Subscriber("front_end/stereo", StereoFrame, self.stereoFrameReceived)
        #self.tracksSub = rospy.Subscriber("front_end_window/FrameTracks", FrameTracks, self.tracksReceived)
        #self.tracksSub = rospy.Subscriber("front_end_window/State", Window, self.windowReceived)
        #self.tracksleftSub = rospy.Subscriber("front_end_window/FrameTracksleft", FrameTracks, self.tracksLeftReceived)
        self.wait=False
        #self.display=imageManager(displayNames)
    def runSimulation(self):
        print("total Settings = ",len(self.extractSettings))
        r = resetDatasetRequest()
        self.resetSrv(r)
        time.sleep(0.5)
        next = publishImageRequest()
        next.command.data = "Next"
        print("beginning Sequence")
        for i in range(5):
            m = setDetectorRequest()
            m.orbConfig = self.extractSettings[i]
            m.Name.data = "ORB"
            m.detection = True
            self.detSrv(m)
            time.sleep(0.15)
            m.detection = False
            self.detSrv(m)
            self.wait=True
            self.publishSrv(next)
            while(self.wait):
                time.sleep(0.05)
        print("completed")
    def stereoFrameReceived(self,message):
        # if("left" in self.display.namesList):
        #     Mat = copy.deepcopy(cv2.cvtColor(self.display.limg,cv2.COLOR_GRAY2RGB))
        #     for index in message.matches:
        #         cv2.circle(Mat, (int(index.leftFeature.imageCoord.x), int(index.leftFeature.imageCoord.y)), 2,(255, 50, 0), -1)
        #     qImg = QImage(Mat.data, Mat.shape[1], Mat.shape[0], 3 * Mat.shape[1], QImage.Format_RGB888)
        #     self.display.updateImage("left",qImg)
        # if("right" in self.display.namesList):
        #     Mat = copy.deepcopy(cv2.cvtColor(self.display.rimg, cv2.COLOR_GRAY2RGB))
        #     for index in message.matches:
        #         cv2.circle(Mat, (int(index.rightFeature.imageCoord.x), int(index.rightFeature.imageCoord.y)), 2,(255, 50, 0), -1)
        #     qImg = QImage(Mat.data, Mat.shape[1], Mat.shape[0], 3 * Mat.shape[1], QImage.Format_RGB888)
        #     self.display.updateImage("right",qImg)
        # print("nLeft",message.nLeft.data)
        #self.stereo.append(message)
        #if(len(self.stereo)>2):
        #    self.stereo.pop(0)
        self.wait=False
    #def tracksReceived(self,message):
        # if((len(self.stereo)>1)and("Tracks" in self.display.namesList)):
        #     drawnImage = copy.deepcopy(cv2.cvtColor(self.display.limg,cv2.COLOR_GRAY2RGB))
        #     for n in range(len(message.currentFrameIndexes)):
        #         pcx = int(self.stereo[1].matches[message.currentFrameIndexes[n].data].leftFeature.imageCoord.x)
        #         pcy = int(self.stereo[1].matches[message.currentFrameIndexes[n].data].leftFeature.imageCoord.y)
        #         prex = int(self.stereo[0].matches[message.previousFrameIndexes[n].data].leftFeature.imageCoord.x)
        #         prey = int(self.stereo[0].matches[message.previousFrameIndexes[n].data].leftFeature.imageCoord.y)
        #         cv2.circle(drawnImage, (pcx, pcy), 2, (0, 255, 0,200), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #         cv2.line(drawnImage, (prex, prey), (pcx, pcy), (0, 255, 0, 200), 1)
        #         cv2.circle(drawnImage, (prex, prey), 2, (0, 255, 0,200), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #     qImg = QImage(drawnImage.data, drawnImage.shape[1], drawnImage.shape[0], 3 * drawnImage.shape[1],QImage.Format_RGB888)
        #     self.display.updateImage("Tracks",qImg)
        #self.wait=False
    def tracksLeftReceived(self,message):
        if((len(self.stereo)>1)and("TracksLeft" in self.display.namesList)):
            drawnImage = copy.deepcopy(cv2.cvtColor(self.display.limg,cv2.COLOR_GRAY2RGB))
            for n in range(len(message.currentFrameIndexes)):
                pcx = int(self.stereo[1].matches[message.currentFrameIndexes[n].data].leftFeature.imageCoord.x)
                pcy = int(self.stereo[1].matches[message.currentFrameIndexes[n].data].leftFeature.imageCoord.y)
                prex = int(self.stereo[0].matches[message.previousFrameIndexes[n].data].leftFeature.imageCoord.x)
                prey = int(self.stereo[0].matches[message.previousFrameIndexes[n].data].leftFeature.imageCoord.y)
                cv2.circle(drawnImage, (pcx, pcy), 2, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                cv2.line(drawnImage, (prex, prey), (pcx, pcy), (255, 255, 0, 200), 1)
                cv2.circle(drawnImage, (prex, prey), 2, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            qImg = QImage(drawnImage.data, drawnImage.shape[1], drawnImage.shape[0], 3 * drawnImage.shape[1],QImage.Format_RGB888)
            self.display.updateImage("TracksLeft",qImg)
        self.wait=False
    def windowReceived(self,message):
        print(len(message.window))
        self.wait=False






if __name__ == '__main__':
    global app
    rospy.init_node('stereoSimulation')
    app = QApplication(sys.argv)
    manager=StereoManager(["TracksLeft"])
    workerThread=Thread(target=manager.runSimulation)
    workerThread.daemon=True
    workerThread.start()

    sys.exit(app.exec_())