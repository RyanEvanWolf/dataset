#!/usr/bin/env python
import rospy
import cv2
from bumbleDataSet import bumbleDataSet
import sys
import time

from dataset.srv import resetDataset,resetDatasetResponse,resetDatasetRequest
from dataset.srv import getDatasetInfo,getDatasetInfoResponse,getDatasetInfoRequest
from dataset.srv import publishImage,publishImageResponse,publishImageRequest
from PyQt5.QtWidgets import QApplication

from std_msgs.msg import Bool
##TODO
###add parameter of max images, otherwise full set

import message_filters
from sensor_msgs.msg import Image


count=0
complete=False
publishSrv = rospy.ServiceProxy("/dataset/publishImage", publishImage)



rospy.init_node('playback_node')
time.sleep(3)
rospy.set_param("endSequence",False)
infoSrv = rospy.ServiceProxy("/dataset/getDatasetInfo", getDatasetInfo)
resetSrv = rospy.ServiceProxy("/dataset/resetDataset", resetDataset)

next = publishImageRequest()
next.command.data = "Next"

r = resetDatasetRequest()
print(resetSrv(r))

print("beginning Sequence")
print(infoSrv(getDatasetInfoRequest()).status.data)



while((not rospy.is_shutdown()) and (not complete)):
    print("Total Counts : ", count)
    count = count + 1
    if (count >= int(sys.argv[1])):
        complete = True
    else:
        time.sleep(1)
        ans = publishSrv(next)
        if (ans.success == False):
            complete = True

#
#
#
#
#
# r = resetDatasetRequest()
# print(resetSrv(r))
# next = publishImageRequest()
# next.command.data = "Next"
# print("beginning Sequence")
# if(len(sys.argv)>=2):
#     print("stopping at image index = ",int(sys.argv[1]))
#     count=0
#     for i in range(int(sys.argv[1])):
#         print(infoSrv(getDatasetInfoRequest()).status.data)
#         publishSrv(next)
#         wait=True
#         print(infoSrv(getDatasetInfoRequest()).status.data)
#         while(wait):
#             time.sleep(0.1)
# else:
#     stop=False
#     while(not stop):
#         wait = True
#         print(infoSrv(getDatasetInfoRequest()).status.data)
#         publishSrv(next)
#         stop=not publishSrv(next).success
#         while(wait):
#             time.sleep(0.1)
#
print("finished sequence playback")
rospy.set_param("endSequence",True)
