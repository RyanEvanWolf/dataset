#!/usr/bin/env python
import os
import cv2
import sys
import struct
import time
import numpy as np
###common ros messages and imports
import rospy
import argparse

parser =argparse.ArgumentParser()
parser.add_argument("rawBagFile")
parser.add_argument("--out",default="/home/ryan/A/")
args=parser.parse_args()

###create directories

if(not os.path.exists(os.path.dirname(args.out))):
    os.makedirs(os.path.dirname(args.out))

print("Reading from File =",args.rawBagFile)
totalBytes=os.path.getsize(args.rawBagFile)
bytesPerImage=2*768*1024

totalImages=int(totalBytes/bytesPerImage)
remainderBytes=totalBytes%bytesPerImage
with open(args.rawBagFile,"rb") as f:
    for imageIndex in range(0,totalImages):
        image=np.zeros((1,bytesPerImage),np.uint8)
        for byteIndex in range(0,bytesPerImage):
            by=struct.unpack("B",f.read(1))
            image[0,byteIndex]=by[0]
        image=image.reshape(2*768,1024)
        #image=cv2.cvtColor(image,cv2.COLOR_BAYER_BG2RGB)
        imageName="%05d.ppm"%int(imageIndex)
        name=args.out+"/"+imageName
        cv2.imwrite(name,image)
        print(imageIndex,totalImages)
        ##copy an image into a np array
    

print("FileInfo--Totalsize=",totalBytes,"--TotalImage=",totalImages,"--remainderBytes=",remainderBytes)


