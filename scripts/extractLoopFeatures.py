#!/usr/bin/env python
#####ros related imports
import rospy 
from front_end.srv import singleImageDetection,singleImageDetectionRequest
from cv_bridge import CvBridge
import os
import cv2
import sys
from front_end.utils import *
import time
import pickle

import argparse

import rosbag

cvb=CvBridge()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("stereo_bag_dir")
    parser.add_argument("output_pickle_dir")
    args=parser.parse_args()

    print("extracting images from : "+args.stereo_bag_dir)

    maxImages=30

    inputBag=rosbag.Bag(args.stereo_bag_dir)
    leftImages=[]
    rightImages=[]

    print("extracting Topic Data")
    for topic,msg,t in inputBag.read_messages(topics=['/bumblebee/left/ROI','/bumblebee/right/ROI']):
        if(topic=="/bumblebee/left/ROI"):
            leftImages.append(msg)
        if(topic=="/bumblebee/right/ROI"):
            rightImages.append(msg)
        print(len(leftImages))
        if((len(leftImages)+len(rightImages))>=2*maxImages):
            break
    inputBag.close()
    newRequest=singleImageDetectionRequest()
    ###get detector Settings
    newRequest.detectorName="FAST"
    newRequest.returnKP=False
    tags,newRequest.det_attrib=getFAST_Attributes()
    ###get descriptor Settings
    #tags2,fakeVar,newRequest.desc_attrib=getSURF_Attributes()
    #newRequest.descriptorName="SURF"

    serviceName="feature_node/singleImageDetection"
    rospy.init_node('extract_loop_features')

    rospy.wait_for_service(serviceName)
    serv=rospy.ServiceProxy(serviceName,singleImageDetection)
    results={}
    results["settings"]=getFAST_Attributes()[1]
    results["data"]=[]
    results["detectorName"]=newRequest.detectorName

    for f in range(0,len(leftImages)):
        newRequest.leftImg=leftImages[f]
        newRequest.rightImg=rightImages[f]
        results["data"].append(serv(newRequest))
        print(f,results["data"][-1].outputFrames[0].nLeft,results["data"][-1].outputFrames[0].nRight)
        print("-----")
    outFile=open(args.output_pickle_dir,"wb")
    print("outFile")
    pickle.dump(results,outFile)
    outFile.close()
    print("completed")