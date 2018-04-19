#!/usr/bin/env python
import os
import cv2
import sys
import struct
import time
import numpy as np
###common ros messages and imports
import rospy

inputDirectory=sys.argv[1]

print("Reading from File =",inputDirectory)
totalBytes=os.path.getsize(inputDirectory)
bytesPerImage=2*768*1024

totalImages=int(totalBytes/bytesPerImage)
remainderBytes=totalBytes%bytesPerImage
with open(inputDirectory,"rb") as f:
    for imageIndex in range(0,totalImages):
        image=np.zeros((1,bytesPerImage),np.uint8)
        for byteIndex in range(0,bytesPerImage):
            by=struct.unpack("B",f.read(1))
            image[0,byteIndex]=by[0]
        image=image.reshape(2*768,1024)
        #image=cv2.cvtColor(image,cv2.COLOR_BAYER_BG2RGB)
        imageName="%05d.ppm"%int(imageIndex)
        name="/home/ryan/DataSets/raw/auto/2/Extracted/"+imageName
        cv2.imwrite(name,image)
        print(imageIndex,totalImages)
        ##copy an image into a np array
    

print("FileInfo--Totalsize=",totalBytes,"--TotalImage=",totalImages,"--remainderBytes=",remainderBytes)


