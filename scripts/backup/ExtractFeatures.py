#!/usr/bin/env python
from FeaturesAnalysis import FeaturesAnalysis
import matplotlib as plt
import rospy
import sys
import cv2
import pickle
rospy.init_node('ExtractFeaturesNode')
if(len(sys.argv)!=3):
    print("invalid input configuration")
    print("should be ->")
    print("rosrun dataset ExtractFeatures.py DetectorName(string) Display(bool)")
else:
    if(sys.argv[2]=="True"):
        #cv2.namedWindow('Features', cv2.WINDOW_NORMAL)
        DataSetDetector=FeaturesAnalysis(str(sys.argv[1]),True)
        #cv2.destroyAllWindows()
    else:
        DataSetDetector = FeaturesAnalysis(str(sys.argv[1]), False)
    data=DataSetDetector.AnalyzeDataSet()
    #pickle.dump(data,open(DataSetDetector.getFullDir()+"/Data.p","wb"))
    #print("saved to " +DataSetDetector.getFullDir())
print("finished")