 
import cv2
import math
import time
import random
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
import copy
from bumblebee.utils import *
from bumblebee.stereo import *
from bumblebee.motion import *
from bumblebee.baseTypes import *
from bumblebee.drawing import *

def MotionCategorySettings():
    Settings={}
    Settings["Fast"]={}
    Settings["Medium"]={}
    Settings["Slow"]={}
    Settings["Fast"]["TranslationMean"]=0.066
    Settings["Fast"]["RotationMean"]=0
    Settings["Fast"]["TranslationNoise"]=0.1*Settings["Fast"]["TranslationMean"] ##meters
    Settings["Fast"]["RotationNoise"]=8      ##degrees

    Settings["Medium"]["TranslationMean"]=0.044
    Settings["Medium"]["RotationMean"]=0
    Settings["Medium"]["TranslationNoise"]=0.1*Settings["Medium"]["TranslationMean"] ##meters
    Settings["Medium"]["RotationNoise"]=4        ##degrees

    Settings["Slow"]["TranslationMean"]=0.022
    Settings["Slow"]["RotationMean"]=0
    Settings["Slow"]["TranslationNoise"]=0.1*Settings["Slow"]["TranslationMean"] ##meters
    Settings["Slow"]["RotationNoise"]=1        ##degrees
    return Settings


def getSimulatedLandmarkSettings():
    Settings={}
    Settings["Xdepth"]=5.0
    Settings["Ydepth"]=5.0
    Settings["Zdepth"]=4.0
    Settings["HeightMaximum"]=0.5
    Settings["MinimumOutlier"]=4.0 #pixels
    Settings["OutlierLevels"]=[0.05,0.1,0.15,0.2,0.25]
    Settings["GaussianNoise"]=[0.25,0.5,0.75,1.0,1.5,2,2.5]
    Settings["operatingCurves"]=[0.3,0.5,0.7,0.9,1.0]
    return Settings


def noisyRotations(noise=5):
    out=np.zeros((1,3))
    out[0,0]=np.random.normal(0,noise,1)
    out[0,1]=np.random.normal(0,noise,1)
    out[0,2]=np.random.normal(0,noise,1)
    return out

def forwardTranslation(zBase=0.2,noise=0.1):
    out=np.zeros((3,1))
    out[0,0]=np.random.normal(0,noise,1)
    out[1,0]=np.random.normal(0,noise,1)
    out[2,0]=abs(np.random.normal(zBase,noise,1))
    return out

def genRandomCoordinate(xAvg,yAvg,zAvg):
    Point=np.ones((4,1),dtype=np.float64)
    Point[0,0]=np.random.normal(0,xAvg,1)
    Point[1,0]=np.random.normal(0,yAvg,1)
    Point[2,0]=np.random.normal(0,zAvg,1)
    return Point

def genGaussianLandmark(camera,simSettings,motionEdge,landmark,gaussian):
    validPoint=False
    output=None
    while(not validPoint):
        verticalNoise=np.random.normal(0,gaussian,1)
        lNoise=np.random.normal(0,gaussian,1)
        rNoise=np.random.normal(0,gaussian,1)
        outAl=copy.deepcopy(landmark[0].L)
        outAl[0,0]+=lNoise
        outAl[1,0]+=verticalNoise
        
        outAr=copy.deepcopy(landmark[0].R)
        outAr[0,0]+=rNoise
        outAr[1,0]=copy.deepcopy(outAl[1,0])


        verticalNoise=np.random.normal(0,gaussian,1)
        lNoise=np.random.normal(0,gaussian,1)
        rNoise=np.random.normal(0,gaussian,1)
        outBl=copy.deepcopy(landmark[1].L)
        outBl[0,0]+=lNoise
        outBl[1,0]+=verticalNoise
        
        outBr=copy.deepcopy(landmark[1].R)
        outBr[0,0]+=rNoise
        outBr[1,0]=copy.deepcopy(outBl[1,0])
        Xa=camera.reproject(outAl,outAr)
        lap,rap=camera.predictPoint(Xa)
        Xb=camera.reproject(outBl,outBr)
        lbp,rbp=camera.predictPoint(Xb)
        if(camera.checkWithinROI(outAl)and camera.checkWithinROI(outAr)
            and camera.checkWithinROI(outBl) and camera.checkWithinROI(outBr)
            and (Xa[1,0]<simSettings["HeightMaximum"])
            and (Xb[1,0]<simSettings["HeightMaximum"])):
            validPoint=True
            output=(stereoEdge(Xa,outAl,outAr,"A"),stereoEdge(Xb,outBl,outBr,"B"))
    return output

def genOutlierLandmark(camera,simSettings,motionEdge,landmark):
    validPoint=False
    output=None
    while(not validPoint):
        ##generate random points
        outAl=np.ones((3,1))
        outAr=np.ones((3,1))
        outBl=np.ones((3,1))
        outBr=np.ones((3,1))
        outAl[0,0]=np.random.uniform(camera.kSettings["roi"][0],
                                camera.kSettings["roi"][0] +camera.kSettings["roi"][2],1)
        outAl[1,0]=np.random.uniform(camera.kSettings["roi"][1],
                                camera.kSettings["roi"][1] +camera.kSettings["roi"][3],1)
        outAr=copy.deepcopy(outAl)
        outAr[0,0]=np.random.uniform(camera.kSettings["roi"][0],
                                camera.kSettings["roi"][0] +camera.kSettings["roi"][2],1)

                                
        outBl[0,0]=np.random.uniform(camera.kSettings["roi"][0],
                                camera.kSettings["roi"][0] +camera.kSettings["roi"][2],1)
        outBl[1,0]=np.random.uniform(camera.kSettings["roi"][1],
                                camera.kSettings["roi"][1] +camera.kSettings["roi"][3],1)
        outBr=copy.deepcopy(outBl)
        outBr[0,0]=np.random.uniform(camera.kSettings["roi"][0],
                                camera.kSettings["roi"][0] +camera.kSettings["roi"][2],1)
        
        Xa=camera.reproject(outAl,outAr)
        lap,rap=camera.predictPoint(Xa)
        Xb=camera.reproject(outBl,outBr)
        lbp,rbp=camera.predictPoint(Xb)
        if((rmsError(outAl,landmark[0].L)>simSettings["MinimumOutlier"])
            and(rmsError(outAr,landmark[0].R)>simSettings["MinimumOutlier"])
            and(rmsError(outBl,landmark[1].L)>simSettings["MinimumOutlier"])
            and(rmsError(outBr,landmark[1].R)>simSettings["MinimumOutlier"])
            and(Xa[1,0]<simSettings["HeightMaximum"])
            and(Xb[1,0]<simSettings["HeightMaximum"])):
            validPoint=True
            output=(stereoEdge(Xa,outAl,outAr,"A"),stereoEdge(Xb,outBl,outBr,"B"))
    return output

def genLandmark(camera,simSettings,motionEdge):
    validPoint=False
    output=None
    while(not validPoint):
        Xa=genRandomCoordinate(simSettings["Xdepth"],
                                  simSettings["Ydepth"],
                                  simSettings["Zdepth"])
        Xb=motionEdge.Htransform.dot(Xa)
        Xb/=Xb[3,0]
        La,Ra=camera.predictPoint(Xa)
        Lb,Rb=camera.predictPoint(Xb)
        if(camera.checkWithinROI(La)and camera.checkWithinROI(Ra)
            and camera.checkWithinROI(Lb) and camera.checkWithinROI(Rb)
            and (Xa[1,0]<simSettings["HeightMaximum"])
            and (Xb[1,0]<simSettings["HeightMaximum"])):
            output=(stereoEdge(Xa,La,Ra,"A"),stereoEdge(Xb,Lb,Rb,"B"))
            # output.tracks.append(stereoEdge(Xa,La,Ra,"A"))
            # output.tracks.append(stereoEdge(Xb,Lb,Rb,"B"))
            validPoint=True
    return output


class motionSimulatedFrame:
    def __init__(self):
        self.motionEdge=None
        self.Points=[]
        self.OperatingCurves={}
    def simulate(self,camera,simSettings,motionEdge,totalLandmarks=5):
        '''
        Gen full set of ideal points

        select the operating curves
        within each operating curve
            ->
        '''
        self.motionEdge=copy.deepcopy(motionEdge)
        self.Points=[]
        self.OperatingCurves={}
        for landmarkIndex in range(0,totalLandmarks):
            singleDataPoint={}
            singleDataPoint["Ideal"]=genLandmark(camera,simSettings,motionEdge)
            singleDataPoint["Outlier"]=genOutlierLandmark(camera,simSettings,motionEdge,singleDataPoint["Ideal"])
            singleDataPoint["Noise"]={}
            for noiseIndex in simSettings["GaussianNoise"]:
                keyName=str(noiseIndex).replace(".","_")
                singleDataPoint["Noise"][keyName]=genGaussianLandmark(camera,simSettings,motionEdge,singleDataPoint["Ideal"],noiseIndex)
            self.Points.append(singleDataPoint)
        ########
        ##determine operating curve selections
        ####
        ##generate outliers, generate selection curves
        for i in simSettings["operatingCurves"]:
            
            keyName=str(int(i*100))
            nFeatures=int(i*totalLandmarks)
            #self.OperatingCurves[keyName]
            currentSamples=sorted(random.sample(range(0,totalLandmarks),nFeatures))
            outlierSelections={}
            for j in simSettings["OutlierLevels"]:
                nOutliers=int(j*nFeatures)
                outlierKeyName=str(int(j*100))
                outlierSelections[outlierKeyName]=sorted(random.sample(range(0,nFeatures),nOutliers))
            self.OperatingCurves[keyName]=(currentSamples,outlierSelections)      
    def getGaussianImage(self,w,h,roi):
        ImageLa=255*np.ones((h,w,3),dtype=np.uint8)
        drawROI(ImageLa,roi)

        ImageLb=copy.deepcopy(ImageLa)
        ImageRa=copy.deepcopy(ImageLa)
        ImageRb=copy.deepcopy(ImageLa)

        idealList=self.getStereoFrame()
        drawTracks(ImageLa,idealList)
        # drawTracks(ImageLa,idealList)
        # drawTracks(ImageLa,idealList)
        # drawTracks(ImageLa,idealList)

        return ImageLa,ImageRa,ImageLb,ImageRb
    def getIdealTracks(self,w,h,roi):
        pass
    #     interF=self.getIdealInterFrameEdge("100")
    #     trackImage=255*np.ones((bumblebee.kSettings["height"],
    #                             bumblebee.kSettings["width"],3),dtype=np.uint8)
    #     drawROI(trackImage,bumblebee.kSettings["roi"])
    #     plotTracks(trackImage,interF.getCurrentL(),interF.getPreviousL())
    #     cv2.namedWindow("Left",cv2.WINDOW_NORMAL)
    #     cv2.imshow("Left",trackImage)        
    # def getOutlierImage(self,w,h,roi):
        pass
    def getStereoFrame(self,name="",args=None):
        if("Outlier"==name):
            return 0
        elif("Noise"==name):
            return 0
        else:
            arrayTracks=[]
            for i in self.Points:
                arrayTracks.append(i["Ideal"])
            return arrayTracks
    def getIdealInterFrameEdge(self,curveName="100"):
        result=interFrameEdge()
        currentSelection=self.OperatingCurves[curveName][0]
        for j in currentSelection:
            result.currentEdges.append(self.Points[j]["Ideal"][1])
            result.previousEdges.append(self.Points[j]["Ideal"][0])
            result.Tracks.append((j,j))
        return result
    # def getIdealTrackCoordinates(self,operatingCurveName):
    #     currentSelection=self.OperatingCurves[operatingCurveName][0]
    #     currentPoints=[]
    #     currentLandmarks=[]
    #     previousPoints=[]
    #     previousLandmarks=[]
    #     for selectedIndex in currentSelection:
    #         Points.append(self.Points[selectedIndex]["Ideal"])
    #     return Points

    #                 selectedSimPoint=currentFile["Points"][pointIndex]
    #         currentPoints.append([selectedSimPoint.Data["Lb"][0,0],
    #                             selectedSimPoint.Data["Lb"][1,0]]) 
    #         currentLandmarks.append(selectedSimPoint.Data["Xb"]) 
    #         previousPoints.append([selectedSimPoint.Data["La"][0,0],
    #                 selectedSimPoint.Data["La"][1,0]])  
    #         previousLandmarks.append(selectedSimPoint.Data["Xa"])