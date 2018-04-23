#!/usr/bin/env python
import os
import cv2
import sys

import time

###common ros messages and imports
import rospy
from std_msgs.msg import String,Int32,Float32
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import pickle

###custom services and messages imports
from dataset.srv import resetDataset,resetDatasetResponse,resetDatasetRequest
from dataset.srv import getDatasetInfo,getDatasetInfoResponse,getDatasetInfoRequest
from dataset.srv import publishImage,publishImageResponse,publishImageRequest


from front_end.srv import singleImageDetection,singleImageDetectionRequest

class loop:
    def __init__(self,data):
        parts=data.split(",")
        self.id=int(parts[1])
        self.startImage=parts[2]
        self.stopImage=parts[3]
    def getImageList(self,rootDir):
        tempFiles=[]
        for files in os.listdir(rootDir):
            tempFiles.append(files)    
        outFiles=sorted(tempFiles)
        startIndex=outFiles.index(self.startImage)
        endIndex=outFiles.index(self.stopImage)
        return outFiles[startIndex:(endIndex+1)]

class singleTrack:
    def __init__(self,dir,id):
        self.dir=dir
        self.id=id
        self.loops=[]
    def printSummary(self):
        out=self.id+"@["+self.dir+"]["
        for i in self.loops:
            out+=str(i.id)
            out+=','
        out+=']'
        return out

class videoSequence:
    def __init__(self,metaInformationDir):
        self.pubImage=rospy.Publisher("dataset/currentImage",Image,queue_size=5)
        self.inDir=metaInformationDir
        f=open(self.inDir,"r")
        ##read header 
        linesRead=0
        self.Track={}
        for line in f:
            parts=line.strip("\n").split(',')
            if(linesRead==0):
                firstTrack=singleTrack(parts[1],parts[0])
                self.Track[firstTrack.id]=firstTrack
                if(len(parts)>2):
                    secondTrack=singleTrack(parts[3],parts[2])
                    self.Track[secondTrack.id]=secondTrack
            else:
                newLoop=loop(line.strip("\n"))
                self.Track[parts[0]].loops.append(newLoop)
            linesRead+=1

        f.close()
    def playbackLoop(self,trackID,loopID,delay=7.5/15.0):
        rootDir=self.Track[trackID].dir
        imageList=self.Track[trackID].loops[loopID-1].getImageList(rootDir)
        cvb=CvBridge()
        for images in imageList:
            print(images+"/"+imageList[-1])
            newImage=cv2.imread(rootDir+"/"+images,cv2.IMREAD_GRAYSCALE)#cv2.cvtColor(cv2.imread(self.data.getCurrentDir(),cv2.IMREAD_GRAYSCALE),cv2.COLOR_BAYER_BG2RGB)
            self.pubImage.publish(cvb.cv2_to_imgmsg(newImage))
            time.sleep(delay)
    def printSummary(self):
        for k in self.Track.keys():
            print(self.Track[k].printSummary())

class bumbleDataSet:
    def __init__(self):
        self.rootDir=None
        self.filesList=[]
        self.currentIndex=-1
    def loadFiles(self,inDir):
        self.rootDir=inDir
        tempFiles=[]
        for files in os.listdir(inDir):
            tempFiles.append(self.rootDir+"/"+files)
        self.filesList=sorted(tempFiles)
        self.currentIndex=-1
    def getImageName(self):
        if(self.currentIndex==-1):
            return "SequenceBeginning"
        elif((self.currentIndex>=0)and(self.currentIndex<len(self.filesList))):
            return self.filesList[self.currentIndex]
        else:
            return "ERROR,INVALID INDEX DETECTED"
    def getStatusString(self):
        return "["+str(self.currentIndex+1)+"/"+str(len(self.filesList))+"]"
    def increment(self):
         if(self.currentIndex+1<len(self.filesList)):
             self.currentIndex+=1
             return True
         else:
             return False
    def decrement(self):
        if(self.currentIndex-1>=0):
            self.currentIndex-=1
            return True
        else:
            return False
    def getStatusString(self):
        return "["+str(self.currentIndex+1)+"/"+str(len(self.filesList))+"]"
    def getCurrentDir(self):
        return self.filesList[self.currentIndex]


class bumbleDataSetNode:
    def __init__(self,rootFolder):
        self.data=bumbleDataSet()
        self.data.loadFiles(rootFolder)
        self.cvb = CvBridge()
        self.infoServ=rospy.Service("dataset/getDatasetInfo",getDatasetInfo,self.info)
        self.pubServ=rospy.Service("dataset/publishImage",publishImage,self.publishLatestImage)
        self.resetServ=rospy.Service("dataset/resetDataset",resetDataset,self.reset)
        self.pubImage=rospy.Publisher("dataset/currentImage",Image,queue_size=5)
    def reset(self,req):
        self.data.currentIndex=-1
        res=resetDatasetResponse()
        return res
    def info(self,req):
        returnMessage=getDatasetInfoResponse()
        returnMessage.rootFolder.data=self.data.rootDir
        returnMessage.currentIndex.data=self.data.currentIndex
        returnMessage.imageName.data=self.data.getImageName()
        returnMessage.status.data=self.data.getStatusString()
        return returnMessage
    def publishLatestImage(self,req):
        res=publishImageResponse()
        if(req.command.data=="Next"):
            if(self.data.increment()):
                newImage=cv2.imread(self.data.getCurrentDir(),cv2.IMREAD_GRAYSCALE)#cv2.cvtColor(cv2.imread(self.data.getCurrentDir(),cv2.IMREAD_GRAYSCALE),cv2.COLOR_BAYER_BG2RGB)
                self.pubImage.publish(self.cvb.cv2_to_imgmsg(newImage))
                res.success=True
            else:
                print("Error, Dataset at end of sequnce")
                res.success=False
        elif(req.command.data=="Current"):
            newImage =cv2.imread(self.data.getCurrentDir(),cv2.IMREAD_GRAYSCALE)# cv2.cvtColor(cv2.imread(self.data.getCurrentDir(), cv2.IMREAD_GRAYSCALE),cv2.COLOR_BAYER_BG2RGB)
            self.pubImage.publish(self.cvb.cv2_to_imgmsg(newImage))
            res.success = True
        elif(req.command.data=="Previous"):
            if (self.data.decrement()):
                newImage = cv2.cvtColor(cv2.imread(self.data.getCurrentDir(), cv2.IMREAD_GRAYSCALE),cv2.COLOR_BAYER_BG2RGB)
                self.pubImage.publish(self.cvb.cv2_to_imgmsg(newImage))
                res.success = True
            else:
                print("Error, Dataset at beginning of Sequence")
                res.success = False
        else:
            print("unrecognized command")
            res.success=False
        res.currentIndex.data=self.data.currentIndex
        res.status.data=self.data.getStatusString()
        return res
if __name__ == '__main__':
    rospy.init_node('dataset')
    print(sys.argv)
    if(len(sys.argv)<2):
        print("Incorrect Parameters Specified")
        print("expected: rosrun dataset bumbleDataSet.py ROOTDIR_OF_DATASET")
        exit(-1)
    manager=bumbleDataSetNode(sys.argv[1])
    rospy.spin()

