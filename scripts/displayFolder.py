#!/usr/bin/env python

import rospy
import cv2
from bumbleDataSet import bumbleDataSet
import sys
import time

rospy.init_node('View')

inData=bumbleDataSet()
inData.loadFiles(sys.argv[1])
debayer=sys.argv[2]
waitTime=int(sys.argv[3])#milliSeconds

cv2.namedWindow("image",cv2.WINDOW_NORMAL)

count=0
for i in inData.filesList:
    
    print(count,i)
    time.sleep(1)
    if(debayer=="True"):
        cv2.imshow("image",cv2.cvtColor(cv2.imread(i,cv2.IMREAD_GRAYSCALE),cv2.COLOR_BAYER_BG2RGB))
    else:
        cv2.imshow("image",cv2.imread(i,cv2.IMREAD_UNCHANGED))
    cv2.waitKey(waitTime)
    count=count+1
    if(rospy.is_shutdown()):
        break
