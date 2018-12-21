import os
import copy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy


def loadImages(imageList):
    return copy.deepcopy([cv2.imread(item,cv2.IMREAD_GRAYSCALE) for item in imageList])


class vidDirectories:
    def __init__(self,sourceData,loopID=None):
        #####
        ##meta data?
        self.root=sourceData
        self.ID=loopID
        self.lImgDir=[]
        self.rImgDir=[]

        print("reading from "+self.root)
        self.lImgDir=sorted(os.listdir(self.root+"/left/rectified"))
        self.rImgDir=sorted(os.listdir(self.root+"/right/rectified"))

        if(self.ID is not None):
            metaDir=self.root+"/meta.txt"
            print("looking for config @ " +metaDir)
            with open(metaDir,"r") as configFile:
                readID,startImg,endImg=configFile.read().splitlines()[loopID-1].split(",")
                print(startImg,endImg)
                sInd=self.lImgDir.index(startImg)
                eInd=self.lImgDir.index(endImg)
                self.lImgDir=self.lImgDir[sInd:eInd+1]
                self.rImgDir=self.rImgDir[sInd:eInd+1]
        print(self.lImgDir[0],self.lImgDir[-1])
        print(self.rImgDir[0],self.rImgDir[-1])
        self.lAbsImgDir = [self.root+"/left/rectified/"+y for y in self.lImgDir]
        self.rAbsImgDir = [self.root+"/right/rectified/"+y for y in self.rImgDir]
        
        print(self.lAbsImgDir[0],self.lAbsImgDir[-1])
        print(self.rAbsImgDir[0],self.rAbsImgDir[-1])
    def getAllImages(self):
        return loadImages(self.lAbsImgDir),loadImages(self.rAbsImgDir)



class vidPublisher(vidDirectories):
    def __init__(self,sourceData,loopID=None,topic="Dataset"):
        vidDirectories.__init__(self,sourceData,loopID)
        self.cvb = CvBridge()
        self.pubName=topic
        self.publ=rospy.Publisher(self.pubName+"/left",Image,queue_size=40)
        self.pubr=rospy.Publisher(self.pubName+"/right",Image,queue_size=40)
    def publishImageIndex(self,index):
        if(index>=0 and index<len(self.lAbsImgDir)):
            self.publ.publish(self.cvb.cv2_to_imgmsg(loadImages([self.lAbsImgDir[index]])[0]))
            self.pubr.publish(self.cvb.cv2_to_imgmsg(loadImages([self.rAbsImgDir[index]])[0]))
