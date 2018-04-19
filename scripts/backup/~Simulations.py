#!/usr/bin/env python
#####ros related imports
import rospy

##required to unpickle the extractionSettings even though they are not directly used
from getFeatures import getOrbParameters

from getFeatures import datasetFeatures,frameFeatures

from front_end.srv import setDetector,setDetectorResponse,setDetectorRequest
from front_end.msg import ORB,StereoFrame,FrameTracks,Window



from dataset.srv import resetDataset,resetDatasetResponse,resetDatasetRequest
from dataset.srv import getDatasetInfo,getDatasetInfoResponse,getDatasetInfoRequest
from dataset.srv import publishImage,publishImageResponse,publishImageRequest

from std_msgs.msg import Bool

import time

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

class Simulation:
    def __init__(self,waitTopic,topicType=Bool,extractFile="/media/ryan/Markov/DetectorOutput/B1/08_01_20-36-44-ORB/devFeatures.p"):
        self.detSrv = rospy.ServiceProxy("front_end/setDetector", setDetector)
        self.publishSrv = rospy.ServiceProxy("/dataset/publishImage", publishImage)
        self.infoSrv = rospy.ServiceProxy("/dataset/getDatasetInfo", getDatasetInfo)
        self.resetSrv = rospy.ServiceProxy("/dataset/resetDataset", resetDataset)
        self.waitSub=rospy.Subscriber(waitTopic,topicType,self.process)
        self.extractSettings = pickle.load(open(extractFile, "rb"))
        self.wait=True
    def process(self,message):
        self.wait=False
    def runSimulation(self):
        print("total Settings = ", len(self.extractSettings))
        r = resetDatasetRequest()
        self.resetSrv(r)
        time.sleep(0.5)
        next = publishImageRequest()
        next.command.data = "Next"
        print("beginning Sequence")
        for i in range(40):
            print(self.infoSrv(getDatasetInfoRequest()).status.data)
            m = setDetectorRequest()
            m.orbConfig = self.extractSettings[i]
            m.Name.data = "ORB"
            m.detection = True
            self.detSrv(m)
            time.sleep(0.15)
            m.detection = False
            self.detSrv(m)
            self.wait = True
            self.publishSrv(next)
            while (self.wait):
                time.sleep(0.05)
        print("completed")


if __name__ == '__main__':
    rospy.init_node('stereoSimulation')
    manager=Simulation("front_end_window/State/updated")
    manager.runSimulation()
    time.sleep(2)
