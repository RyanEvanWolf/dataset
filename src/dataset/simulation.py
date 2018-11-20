 
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
from bumblebee.camera import *


def MotionCategorySettings():
    Settings={}
    Settings["Fast"]={}
    Settings["Medium"]={}
    Settings["Slow"]={}
    Settings["Fast"]["TranslationMean"]=0.066
    Settings["Fast"]["RotationMean"]=30
    Settings["Fast"]["TranslationNoise"]=0.1*Settings["Fast"]["TranslationMean"] ##meters
    Settings["Fast"]["RotationNoise"]=8    ##degrees

    Settings["Medium"]["TranslationMean"]=0.044
    Settings["Medium"]["RotationMean"]=20
    Settings["Medium"]["TranslationNoise"]=0.1*Settings["Medium"]["TranslationMean"] ##meters
    Settings["Medium"]["RotationNoise"]=4        ##degrees

    Settings["Slow"]["TranslationMean"]=0.022
    Settings["Slow"]["RotationMean"]=10
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
    Settings["OutlierLevels"]=[0.05,0.12,0.2]
    Settings["GaussianNoise"]=[0.25,0.75,1.25]
    Settings["operatingCurves"]=[0.1,0.4,0.7,1.0]
    return Settings

def genDefaultNisterSettings(cameraConfig):
    settings={}
    settings["Pl"]=cameraConfig["Pl"]
    settings["Pr"]=cameraConfig["Pr"]
    settings["pp"]=cameraConfig["pp"]
    settings["k"]=cameraConfig["k"]
    settings["f"]=cameraConfig["f"]
    settings["threshold"]=3
    settings["probability"]=0.99
    return settings

def noisyRotations(noise=5):
    out=np.zeros((3,1))
    out[0,0]=np.random.normal(0,noise,1)
    out[1,0]=np.random.normal(0,noise,1)
    out[2,0]=np.random.normal(0,noise,1)
    return out

def forwardTranslation(zBase=0.2,noise=0.1):
    out=np.zeros((3,1))
    out[0,0]=np.random.normal(0,noise,1)
    out[1,0]=np.random.normal(0,noise,1)
    out[2,0]=abs(np.random.normal(zBase,noise,1))
    return out

def dominantRotation(yawBase=15,noise=5):
    out=np.zeros((3,1))
    out[0,0]=np.random.normal(0,noise,1)
    out[1,0]=np.random.normal(0,noise,1)
    out[2,0]=np.clip(abs(np.random.normal(yawBase,noise,1)),0,40)   
    return out

def genRandomCoordinate(xAvg,yAvg,zAvg):
    Point=np.ones((4,1),dtype=np.float64)
    Point[0,0]=np.random.normal(0,xAvg,1)
    Point[1,0]=np.random.normal(0,yAvg,1)
    Point[2,0]=np.random.normal(0,zAvg,1)
    return Point

def genGaussianLandmark(camera,simSettings,landmark,gaussian):
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
        if(camera.checkWithinROI(outAl)and camera.checkWithinROI(outAr,False)
            and camera.checkWithinROI(outBl) and camera.checkWithinROI(outBr,False)
            and (Xa[1,0]<simSettings["HeightMaximum"])
            and (Xb[1,0]<simSettings["HeightMaximum"])):
            validPoint=True
            output=(stereoEdge(Xa,outAl,outAr,"A"),stereoEdge(Xb,outBl,outBr,"B"))
    return output

def genOutlierLandmark(camera,simSettings,landmark):
    validPoint=False
    output=None
    while(not validPoint):
        ##generate random points
        outAl=np.ones((3,1))
        outAr=np.ones((3,1))
        outBl=np.ones((3,1))
        outBr=np.ones((3,1))
        outAl[0,0]=np.random.uniform(camera.kSettings["lInfo"].roi.x_offset,
                                camera.kSettings["lInfo"].roi.x_offset +camera.kSettings["lInfo"].roi.width,1)
        outAl[1,0]=np.random.uniform(camera.kSettings["lInfo"].roi.y_offset,
                                camera.kSettings["lInfo"].roi.y_offset +camera.kSettings["lInfo"].roi.height,1)
        outAr=copy.deepcopy(outAl)

        outAr[0,0]=np.random.uniform(camera.kSettings["rInfo"].roi.x_offset,
                        camera.kSettings["rInfo"].roi.x_offset +camera.kSettings["rInfo"].roi.width,1)

        outBl[0,0]=np.random.uniform(camera.kSettings["lInfo"].roi.x_offset,
                                camera.kSettings["lInfo"].roi.x_offset +camera.kSettings["lInfo"].roi.width,1)
        outBl[1,0]=np.random.uniform(camera.kSettings["lInfo"].roi.y_offset,
                                camera.kSettings["lInfo"].roi.y_offset +camera.kSettings["lInfo"].roi.height,1)
        outBr=copy.deepcopy(outBl)
        outBr[0,0]=np.random.uniform(camera.kSettings["rInfo"].roi.x_offset,
                        camera.kSettings["rInfo"].roi.x_offset +camera.kSettings["rInfo"].roi.width,1)                        

        
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

def genLandmark(camera,simSettings,pose):
    validPoint=False
    output=None
    count=0
    while(not validPoint):
        Xa=genRandomCoordinate(simSettings["Xdepth"],
                                  simSettings["Ydepth"],
                                  simSettings["Zdepth"])    
        Xb=pose.getH().dot(Xa)
        Xb/=Xb[3,0]

        test=np.linalg.inv(pose.getH()).dot(Xb)
        test/=test[3,0]

        La,Ra=camera.predictPoint(Xa)
        Lb,Rb=camera.predictPoint(Xb)


        if(camera.checkWithinROI(La)and camera.checkWithinROI(Ra,False)
            and camera.checkWithinROI(Lb) and camera.checkWithinROI(Rb,False)
            and (Xa[1,0]<simSettings["HeightMaximum"])
            and (Xb[1,0]<simSettings["HeightMaximum"])):
            output=(stereoEdge(Xa,La,Ra,"A"),stereoEdge(Xb,Lb,Rb,"B"))
            validPoint=True
    return output

def ROIcheck(pt,roi):
    ###
    ##check width
    if((pt[0,0]>=roi[0])and(pt[0,0]<=(roi[0]+roi[2]))):
        if((pt[1,0]>=roi[1])and(pt[1,0]<=(roi[1]+roi[3]))):   
            return True
        else:
            return False
    else:
        return False


class outlierWindow(slidingWindow):
    def __init__(self,Ksettings):
        super(outlierWindow,self).__init__(cameraSettings=Ksettings)
        pass

class noisyWindow(slidingWindow):
    def __init__(self,Ksettings):
        super(noisyWindow,self).__init__(cameraSettings=Ksettings)
        self.OperatingCurves={}
        self.noisyWindows={}    
    def simulate(self,idealData,lSettings):

        self.OperatingCurves=copy.deepcopy(idealData.OperatingCurves)
        for opCurve in self.OperatingCurves.keys():
            currentN=len(self.OperatingCurves[opCurve])
            print(currentN)
            currentNoiseW=idealWindow(self.kSettings)
            currentNoiseW.X=np.zeros((6+currentN*4),1)
            currentNoiseW.M=[]
            currentNoiseW.nLandmarks=currentN
            self.tracks=[]
            #populate the X and M vectors
            


        # self.X=copy.deepcopy(idealDa
            for noise in lSettings["GaussianNoise"]:
                print(noise)

        # self.M=copy.deepcopy(idealData.M)
        # self.tracks=copy.deepcopy(idealData.tracks)
        # self.inliers=None
        # self.nLandmarks=
        # self.nPoses=frames
        # abcd=None

class idealWindow(slidingWindow):
    def __init__(self,Ksettings):
        super(idealWindow,self).__init__(cameraSettings=Ksettings)
        self.motion=np.zeros((6,1))
        self.OperatingCurves={}
    def simulate(self,lSettings,Rtheta,C,nPoints=500):
        '''
        lSettings= landmark simulation settings
        Rtheta= [roll pitch yaw]^T (in degrees)
        C  = metric distance such that P=[R|-RC]=[R|T]
        '''
        self.motion[0:3,0]=np.radians(Rtheta).reshape(3)
        R=composeR(self.motion[0:3,0])
        self.motion[3:6,0]=-R.dot(C).flatten()
        self.nLandmarks=nPoints
        self.X=np.zeros((6 + nPoints*4,1))
        self.X[0:6,0]=self.motion.flatten()
        self.tracks=[]
        for i in range(nPoints):
            self.tracks.append(range(self.nPoses))
            validPoint=False
            while(not validPoint):
                Xa=genRandomCoordinate(lSettings["Xdepth"],
                                        lSettings["Ydepth"],
                                        lSettings["Zdepth"])    
                Xb=getH(self.motion).dot(Xa)
                Xb/=Xb[3,0]
                La,Ra=self.kSettings["Pl"].dot(Xa),self.kSettings["Pr"].dot(Xa)
                Lb,Rb=self.kSettings["Pl"].dot(Xb),self.kSettings["Pr"].dot(Xb)
                La/=La[2,0]
                Ra/=Ra[2,0]
                Lb/=Lb[2,0]
                Rb/=Rb[2,0]

                if(self.checkWithinROI(La)and self.checkWithinROI(Ra,False)
                    and self.checkWithinROI(Lb) and self.checkWithinROI(Rb,False)
                    and (Xa[1,0]<lSettings["HeightMaximum"])
                    and (Xb[1,0]<lSettings["HeightMaximum"])):
                    validPoint=True
                    #########
                    ##pack into Frame
                    self.X[6 +4*i:6+4*i+4,0]=Xa.reshape(4)
                    M=np.zeros((4,2))
                    M[0:2,0]=La[0:2,0]#,Ra[0:2,0]))
                    M[2:4,0]=Ra[0:2,0]
                    M[0:2,1]=Lb[0:2,0]#,Ra[0:2,0]))
                    M[2:4,1]=Rb[0:2,0]
                    self.M.append(M)
        #######
        ##define simulation curves
         #generate outliers, generate selection curves
        for i in lSettings["operatingCurves"]:
            keyName=str(int(i*100))
            nFeatures=int(i*nPoints)
            #self.OperatingCurves[keyName]
            currentSamples=sorted(random.sample(range(0,nPoints),nFeatures))
            self.OperatingCurves[keyName]=currentSamples
    def checkWithinROI(self,pt,left=True):
        if(left):
            return ROIcheck(pt,ROIfrmMsg(self.kSettings["lInfo"].roi))
        else:
            return ROIcheck(pt,ROIfrmMsg(self.kSettings["rInfo"].roi))      




class simulatedDataFrame:
    def __init__(self,cameraType="subROI",configurationTopic=""):
        self.kSettings={}
        self.OperatingCurves={}
        if(configurationTopic==""):
            self.kSettings=getCameraSettingsFromServer(cameraType=cameraType)
        else:
            self.kSettings=getCameraSettingsFromServer(configurationTopic,cameraType)
    
        self.idealMotion=np.zeros((6,1))   #in H=[R|T] shape (Metric= [R|-RC])
        self.idealWindow=slidingWindow(self.kSettings)
        self.Gaussian={}
        self.Outlier={}
    ################
    ##admin
    ################
    def serializeFrame(self):
        binDiction={}
        binDiction["kSettings"]=pickle.dumps(self.kSettings)
        binDiction["OperatingCurves"]=pickle.dumps(self.OperatingCurves)
        binDiction["idealWindow"]=self.idealWindow.serializeWindow()
        binDiction["idealMotion"]=pickle.dumps(self.idealMotion)
        binDiction["Gaussian"]={}
        binDiction["Outlier"]={}
        for i in self.Gaussian.keys():
            binDiction["Gaussian"][i]=self.Gaussian[i].serializeWindow()
        for i in self.Outlier.keys():
            binDiction["Outlier"][i]={}
            for j in self.Outlier[i].keys():
                binDiction["Outlier"][i][j]={}
                binDiction["Outlier"][i][j]["Outliers"]=pickle.dumps(self.Outlier[i][j]["Outliers"])
                binDiction["Outlier"][i][j]["Inliers"]=pickle.dumps(self.Outlier[i][j]["Inliers"])
                binDiction["Outlier"][i][j]["data"]=self.Outlier[i][j]["data"].serializeWindow()
        return msgpack.dumps(binDiction)
    def deserializeFrame(self,data):
        intern=msgpack.loads(data)
        self.kSettings=pickle.loads(intern["kSettings"])
        self.OperatingCurves=pickle.loads(intern["OperatingCurves"])
        self.idealWindow=slidingWindow(pickle.loads(intern["kSettings"]))
        self.idealWindow.deserializeWindow(intern["idealWindow"])
        self.idealMotion=pickle.dumps(intern["idealMotion"])
        self.Gaussian={}
        for i in intern["Gaussian"].keys():
            self.Gaussian[i]=slidingWindow(pickle.loads(intern["kSettings"]))
            self.Gaussian[i].deserializeWindow(intern["Gaussian"][i])
        self.Outlier={}
        for i in intern["Outlier"].keys():
            self.Outlier[i]={}
            for j in intern["Outlier"][i].keys():
                self.Outlier[i][j]={}
                self.Outlier[i][j]["Outliers"]=pickle.loads(intern["Outlier"][i][j]["Outliers"])
                self.Outlier[i][j]["Inliers"]=pickle.loads(intern["Outlier"][i][j]["Inliers"])
                self.Outlier[i][j]["data"]=slidingWindow(pickle.loads(intern["kSettings"]))
                self.Outlier[i][j]["data"].deserializeWindow(intern["Outlier"][i][j]["data"])
    def runSimulation(self,lSettings,Rtheta,C,nPoints=500):
        '''
        lSettings= landmark simulation settings
        Rtheta= [roll pitch yaw]^T (in degrees)
        C  = metric distance such that P=[R|-RC]=[R|T]
        '''
        #########
        ##generate Ideal Motion
        self.idealMotion[0:3,0]=np.radians(Rtheta).reshape(3)
        R=composeR(self.idealMotion[0:3,0])
        self.idealMotion[3:6,0]=-R.dot(C).flatten()
        self.idealWindow.nLandmarks=nPoints
        self.idealWindow.X=np.zeros((6 + nPoints*4,1))
        self.idealWindow.X[0:6,0]=self.idealMotion.flatten()
        self.idealWindow.tracks=[]
        for i in range(nPoints):
            self.idealWindow.tracks.append(range(self.idealWindow.nPoses))
            validPoint=False
            while(not validPoint):
                Xa=genRandomCoordinate(lSettings["Xdepth"],
                                        lSettings["Ydepth"],
                                        lSettings["Zdepth"])    
                Xb=getH(self.idealMotion).dot(Xa)
                Xb/=Xb[3,0]
                La,Ra=self.kSettings["Pl"].dot(Xa),self.kSettings["Pr"].dot(Xa)
                Lb,Rb=self.kSettings["Pl"].dot(Xb),self.kSettings["Pr"].dot(Xb)
                La/=La[2,0]
                Ra/=Ra[2,0]
                Lb/=Lb[2,0]
                Rb/=Rb[2,0]

                if(self.checkWithinROI(La)and self.checkWithinROI(Ra,False)
                    and self.checkWithinROI(Lb) and self.checkWithinROI(Rb,False)
                    and (Xa[1,0]<lSettings["HeightMaximum"])
                    and (Xb[1,0]<lSettings["HeightMaximum"])):
                    validPoint=True
                    #########
                    ##pack into Frame
                    self.idealWindow.X[6 +4*i:6+4*i+4,0]=Xa.reshape(4)
                    M=np.zeros((4,2))
                    M[0:2,0]=La[0:2,0]#,Ra[0:2,0]))
                    M[2:4,0]=Ra[0:2,0]
                    M[0:2,1]=Lb[0:2,0]#,Ra[0:2,0]))
                    M[2:4,1]=Rb[0:2,0]
                    self.idealWindow.M.append(M)
        #######
        ##define simulation curves
        for i in lSettings["operatingCurves"]:
            keyName=str(int(i*100))
            nFeatures=int(i*nPoints)
            currentSamples=sorted(random.sample(range(0,nPoints),nFeatures))
            self.OperatingCurves[keyName]=currentSamples
            self.Outlier[keyName]={}
            for j in lSettings["OutlierLevels"]:
                nOutliers=int(j*nFeatures)
                outlierKeyName=str(int(j*100))
                self.Outlier[keyName][outlierKeyName]={}
                self.Outlier[keyName][outlierKeyName]["Outliers"]=sorted(random.sample(range(0,nFeatures),nOutliers))
                self.Outlier[keyName][outlierKeyName]["Inliers"]=list(set(range(0,nFeatures))-set(self.Outlier[keyName][outlierKeyName]["Outliers"]))        #####
        ##add Gaussian noise curves
        for noise in lSettings["GaussianNoise"]:
            keyName=str(noise)
            self.Gaussian[keyName]=copy.deepcopy(self.idealWindow)
            #####
            ##add Gaussian Noise onto projections
            for m in range(0,nPoints):
                validNoise=False
                while(not validNoise):
                    testPts=copy.deepcopy(self.Gaussian[keyName].M[m])+np.random.normal(0,noise,self.Gaussian[keyName].M[m].shape)
                    if( self.checkWithinROI(testPts[0:2,0].reshape(2,1))
                        and self.checkWithinROI(testPts[2:4,0].reshape(2,1),left=False)
                        and self.checkWithinROI(testPts[0:2,1].reshape(2,1))
                        and self.checkWithinROI(testPts[2:4,1].reshape(2,1),left=False)):
                        validNoise=True
                        self.Gaussian[keyName].M[m]=testPts     
        #####
        ##generate outlier measurements
        moutliers=[]

        for i in self.idealWindow.M:
            outlierM=copy.deepcopy(i)

            moutliers.append(outlierM)
        for i in lSettings["operatingCurves"]:

            keyName=str(int(i*100))
            for outlier in lSettings["OutlierLevels"]:
                outlierKeyName=str(int(outlier*100))
                self.Outlier[keyName][outlierKeyName]["data"]=copy.deepcopy(self.idealWindow.getSubset(self.OperatingCurves[keyName]))
                for outIndex in self.Outlier[keyName][outlierKeyName]["Outliers"]:

                    lU=np.random.uniform(self.kSettings["lInfo"].roi.x_offset,
                                            self.kSettings["lInfo"].roi.x_offset+
                                            self.kSettings["lInfo"].roi.width,2)
                    lv=np.random.uniform(self.kSettings["lInfo"].roi.y_offset,
                                            self.kSettings["lInfo"].roi.y_offset+
                                            self.kSettings["lInfo"].roi.height,2)

                    rU=np.random.uniform(self.kSettings["rInfo"].roi.x_offset,
                                            self.kSettings["rInfo"].roi.x_offset+
                                            self.kSettings["rInfo"].roi.width,2)
                    rv=np.random.uniform(self.kSettings["rInfo"].roi.y_offset,
                                            self.kSettings["rInfo"].roi.y_offset+
                                            self.kSettings["rInfo"].roi.height,2)
                    outlierM[0,0]=lU[0]
                    outlierM[1,0]=lv[0]
                    outlierM[2,0]=rU[0]
                    outlierM[3,0]=rv[0]


                    outlierM[0,1]=lU[1]
                    outlierM[1,1]=lv[1]
                    outlierM[2,1]=rU[1]
                    outlierM[3,1]=rv[1]
                    self.Outlier[keyName][outlierKeyName]["data"].M[outIndex]=outlierM
        #     #####
    def checkWithinROI(self,pt,left=True):
        if(left):
            return ROIcheck(pt,ROIfrmMsg(self.kSettings["lInfo"].roi))
        else:
            return ROIcheck(pt,ROIfrmMsg(self.kSettings["rInfo"].roi))          
# class motionSimulatedFrame:
#     def __init__(self):
#         self.pose=None
#         self.Points=[]
#         self.OperatingCurves={}
#     def simulate(self,camera,simSettings,pose,totalLandmarks=5):
#         '''
#         Gen full set of ideal points

#         select the operating curves
#         within each operating curve
#             ->
#         '''
#         self.pose=copy.deepcopy(pose)
#         self.Points=[]
#         self.OperatingCurves={}
#         for landmarkIndex in range(0,totalLandmarks):
#             singleDataPoint={}
#             singleDataPoint["Ideal"]=genLandmark(camera,simSettings,self.pose)
#             singleDataPoint["Outlier"]=genOutlierLandmark(camera,simSettings,singleDataPoint["Ideal"])
#             singleDataPoint["Noise"]={}
#             for noiseIndex in simSettings["GaussianNoise"]:
#                 keyName=str(noiseIndex).replace(".","_")
#                 singleDataPoint["Noise"][keyName]=genGaussianLandmark(camera,simSettings,singleDataPoint["Ideal"],noiseIndex)
#             self.Points.append(singleDataPoint)
#     #     ########
#     #     ##determine operating curve selections
#     #     ####
#         ##generate outliers, generate selection curves
#         for i in simSettings["operatingCurves"]:
            
#             keyName=str(int(i*100))
#             nFeatures=int(i*totalLandmarks)
#             #self.OperatingCurves[keyName]
#             currentSamples=sorted(random.sample(range(0,totalLandmarks),nFeatures))
#             outlierSelections={}
#             for j in simSettings["OutlierLevels"]:
#                 nOutliers=int(j*nFeatures)
#                 outlierKeyName=str(int(j*100))
#                 outlierSelections[outlierKeyName]=sorted(random.sample(range(0,nFeatures),nOutliers))
#             self.OperatingCurves[keyName]=(currentSamples,outlierSelections)      
#     def getGaussianImage(self,w,h,roi):
#         ImageLa=255*np.ones((h,w,3),dtype=np.uint8)
#         drawROI(ImageLa,roi)

#         ImageLb=copy.deepcopy(ImageLa)
#         ImageRa=copy.deepcopy(ImageLa)
#         ImageRb=copy.deepcopy(ImageLa)

#         idealList=self.getStereoFrame()
#         drawTracks(ImageLa,idealList)
#         # drawTracks(ImageLa,idealList)
#         # drawTracks(ImageLa,idealList)
#         # drawTracks(ImageLa,idealList)

#         return ImageLa,ImageRa,ImageLb,ImageRb
#     def getStereoFrame(self,name="",args=None):
#         if("Outlier"==name):
#             return 0
#         elif("Noise"==name):
#             return 0
#         else:
#             arrayTracks=[]
#             for i in self.Points:
#                 arrayTracks.append(i["Ideal"])
#             return arrayTracks
#     def getIdealInterFrameEdge(self,curveName="100"):
#         result=interFrameEdge()
#         currentSelection=self.OperatingCurves[curveName][0]
#         for j in currentSelection:
#             result.currentEdges.append(self.Points[j]["Ideal"][1])
#             result.previousEdges.append(self.Points[j]["Ideal"][0])
#             result.Tracks.append((j,j))
#         return result
#     def getOutlierInterFrameEdge(self,curveName,outlierName):
#         result=interFrameEdge()
#         currentSelection=self.OperatingCurves[curveName][0]
#         outlierSelection=self.OperatingCurves[curveName][1][outlierName]
#         outCount=0
#         for j in currentSelection:
#             if(j in outlierSelection):
#                 result.currentEdges.append(self.Points[j]["Outlier"][1])
#                 result.previousEdges.append(self.Points[j]["Outlier"][0])
#                 result.Tracks.append((j,j))
#                 outCount+=1
#             else:
#                 result.currentEdges.append(self.Points[j]["Ideal"][1])
#                 result.previousEdges.append(self.Points[j]["Ideal"][0])
#                 result.Tracks.append((j,j))         
#         return result
#     def getNoisyInterFrameEdge(self,curveName,noiseName):
#         result=interFrameEdge()
#         currentSelection=self.OperatingCurves[curveName][0]
#         for j in currentSelection:
#             result.currentEdges.append(self.Points[j]["Noise"][noiseName][1])
#             result.previousEdges.append(self.Points[j]["Noise"][noiseName][0])
#             result.Tracks.append((j,j))
#         return result
