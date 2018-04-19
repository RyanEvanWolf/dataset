import rospy
from cv_bridge import CvBridge
import os
import time
##ros general imports
from std_msgs.msg import Int32, Bool, Int8, Float64,String

##camera settings
from dataset.srv import rectifiedSettings,rectifiedSettingsResponse,rectifiedSettingsRequest

##detector services
from dataset.srv import extractFAST, extractFASTResponse,extractFASTRequest
from dataset.srv import extractSIFT,extractSIFTResponse,extractSIFTRequest
from dataset.srv import extractORB,extractORBResponse,extractORBRequest
from dataset.srv import extractSURF,extractSURFResponse,extractSURFRequest
from dataset.srv import extractAKAZE,extractAKAZEResponse,extractAKAZERequest
from dataset.srv import extractBRISK,extractBRISKResponse,extractBRISKRequest


from dataset.msg import FAST, SIFT, SURF, AKAZE, BRISK
from dataset.srv import resetDataset,resetDatasetResponse,resetDatasetRequest
from dataset.srv import getDatasetInfo,getDatasetInfoResponse,getDatasetInfoRequest
from dataset.srv import publishImage,publishImageResponse,publishImageRequest


from front_end.msg import ORB
from front_end.srv import setDetector,setDetectorRequest,setDetectorResponse
from front_end.srv import detectCurrent,detectCurrentRequest,detectCurrentResponse
from drawingFunctions import ImageThread
import pickle

import numpy as np
import cv2
import copy
#post processing imports
import statistics
import matplotlib.pyplot as plt
import matplotlib.style as sty

from sensor_msgs.msg import Image

import datetime


def getOrbParameters():
    ORB_Messages = []
    scaleVect =np.linspace(1.0, 2.5, 4, endpoint=True)#,8
    edgeVect = np.arange(2,32,10)#np.arange(2, 32,2)
    levelVect = np.arange(2, 10, 4)#2,10,2
    wtaVect = np.arange(2, 4, 1)
    scoreVect = [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]
    patchVect =np.arange(32, 64, 32)#16,64,16
    for sc in scaleVect:
        for pat in patchVect:
            for ed in edgeVect:
                for wt in wtaVect:
                    for l in levelVect:
                        for scor in scoreVect:
                            newSettings = ORB()
                            newSettings.maxFeatures.data=10000
                            newSettings.scale.data = sc
                            newSettings.edge.data = ed
                            newSettings.level.data = l
                            newSettings.wta.data = wt
                            newSettings.score.data = scor
                            newSettings.patch.data = pat
                            ORB_Messages.append(newSettings)
    return ORB_Messages

def getKAZEParameters():
    KAZE_Messages=[]
    offset=[1.60]
    omax=[4]
    nsublevels=[4]
    dthresh=[0.0003]
    min_dthreshold=[0.000001]
    use_fed=[True]
    descriptor=[0]
    diffusivity=[0]
    sderivatives=[1.0]
    kcontrast=[0.001]
    kcontrast_percentile=[0.7]
    kcontrast_bins=[300]
    for off in offset:
        for m in omax:
            for subl in nsublevels:
                for thresh in dthresh:
                    for mi in min_dthreshold:
                        for use in use_fed:
                            for desc in descriptor:
                                for diff in diffusivity:
                                    for s in sderivatives:
                                        for kcon in kcontrast:
                                            for kper in kcontrast_percentile:
                                                for kbin in kcontrast_bins:
                                                    newSettings=KAZE()
                                                    newSettings.offset.data=off
                                                    newSettings.omax.data=m
                                                    newSettings.sublevels.data=subl
                                                    newSettings.dthreshold.data=thresh
                                                    newSettings.minthresh.data=mi
                                                    newSettings.use_fed.data=use
                                                    newSettings.descriptor.data=desc
                                                    newSettings.diffusivity.data=diff
                                                    newSettings.sderivative.data=s
                                                    newSettings.kcontrast.data=kcon
                                                    newSettings.kcontrast_percentile.data=kper
                                                    newSettings.kcontrast_percentile.data=kbin
                                                    KAZE_Messages.append(newSettings)
    return KAZE_Messages




def getSURFParameters():
    SURF_Messages=[]
    threshold=np.linspace(0.8, 2.5, 5, endpoint=True)
    nOctaves=np.arange(2, 6,2)
    nOctavesLayers=np.arange(2,6,2)
    extended=[False,True]
    upright=[False,True]
    for t in threshold:
        for oct in nOctaves:
            for layers in nOctavesLayers:
                for e in extended:
                    for u in upright:
                        newSettings=SURF()
                        newSettings.thresh.data=t
                        newSettings.nOctave.data=oct
                        newSettings.nOctaveLayer.data=layers
                        newSettings.extended.data=e
                        newSettings.upright.data=u
                        SURF_Messages.append(newSettings)
    return SURF_Messages

def getFastParameters():
    FAST_Messages=[]
    threshVect=np.arange(4, 8, 2)
    typeVect=[cv2.FAST_FEATURE_DETECTOR_TYPE_5_8, cv2.FAST_FEATURE_DETECTOR_TYPE_7_12, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16]
    SupprVect=[True]#, False]
    for s in SupprVect:
        for ty in typeVect:
            for th in threshVect:
                fastSettings=FAST()
                fastSettings.type.data=ty
                fastSettings.thresh.data=th
                fastSettings.suppression.data=s
                FAST_Messages.append(fastSettings)
    return FAST_Messages

def getBRISKParameters():
    BRISK_Messages=[]
    thresh_vect=np.arange(4, 8, 2)
    oct_vect=[2]
    scale_vect=[1.0,0.9]
    for t in thresh_vect:
        for o in oct_vect:
            for s in scale_vect:
                newSettings=BRISK()
                newSettings.thresh.data=t
                newSettings.octaves.data=o
                newSettings.patternScale.data=s
                BRISK_Messages.append(newSettings)
    return BRISK_Messages
def getSIFTParameters():
    SIFT_Messages=[]
    nFeatures=10000
    octave=np.arange(2, 3, 1)
    contrast=np.linspace(0.001, 0.5, 2, endpoint=True)
    edge=np.linspace(1, 4, 2, endpoint=True)
    sigma=np.linspace(1, 2, 2, endpoint=True)
    for o in octave:
        for c in contrast:
            for e in edge:
                for s in sigma:
                    SIFTsettings=SIFT()
                    SIFTsettings.maxFeatures.data=nFeatures
                    SIFTsettings.nOctave.data=o
                    SIFTsettings.contrastThresh.data=c
                    SIFTsettings.edgeThresh.data=e
                    SIFTsettings.sigma.data=s
                    SIFT_Messages.append(SIFTsettings)
    return SIFT_Messages

def getAKAZEParameters():
    AKAZE_Messages=[]
    offset = [1.60]
    derFactor=[1.5]
    omax = [4]
    nsublevels = [4]
    dthresh = [0.0003]
    min_dthreshold = [0.000001]
    patternSize = [10]
    descriptor = [0]  #0-5
    diffusivity = [0] #0-3
    sderivatives = [1.0]
    kcontrast = [0.001]
    kcontrast_percentile = [0.7]
    kcontrast_bins =[300]
    for off in offset:
        for der in derFactor:
            for m in omax:
                for subl in nsublevels:
                    for thresh in dthresh:
                        for mi in min_dthreshold:
                            for pat in patternSize:
                                for desc in descriptor:
                                    for diff in diffusivity:
                                        for s in sderivatives:
                                            for kcon in kcontrast:
                                                for kper in kcontrast_percentile:
                                                    for kbin in kcontrast_bins:
                                                        newSettings = AKAZE()
                                                        newSettings.offset.data = off
                                                        newSettings.derivative_factor.data=der
                                                        newSettings.omax.data = m
                                                        newSettings.sublevels.data = subl
                                                        newSettings.dthreshold.data = thresh
                                                        newSettings.minthresh.data = mi
                                                        newSettings.patternSize.data=pat
                                                        newSettings.descriptor.data = desc
                                                        newSettings.diffusivity.data = diff
                                                        newSettings.sderivative.data = s
                                                        newSettings.kcontrast.data = kcon
                                                        newSettings.kcontrast_percentile.data = kper
                                                        newSettings.kcontrast_nbins.data = kbin
                                                        AKAZE_Messages.append(newSettings)
    return AKAZE_Messages

from bumbleDataSet import bumbleDataSetNode



# class FeaturesAnalysis:
#     def __init__(self, detectorName, display=True):
#         self.s=ImageThread("det")
#         try:
#             rospy.wait_for_service("/dataset/publishImage", 4)
#             print("Connected to /dataset/publishImage")
#             self.publishSrv = rospy.ServiceProxy("/dataset/publishImage",publishImage)
#         except rospy.ServiceException as exc:
#             print("Service did not process request: " + str(exc))
#         try:
#             rospy.wait_for_service("/dataset/getDatasetInfo", 4)
#             print("Connected to /dataset/getDatasetInfo")
#             self.infoSrv = rospy.ServiceProxy("/dataset/getDatasetInfo",getDatasetInfo)
#         except rospy.ServiceException as exc:
#             print("Service did not process request: " + str(exc))
#         try:
#             rospy.wait_for_service("/dataset/resetDataset", 4)
#             print("Connected to /dataset/resetDataset")
#             self.resetSrv = rospy.ServiceProxy("/dataset/resetDataset", resetDataset)
#         except rospy.ServiceException as exc:
#             print("Service did not process request: " + str(exc))
#         self.startTime=time.strftime('%d_%m_%H_%M_%S')
#         self.detectorName=detectorName
#         self.wait=False
#         self.settings=None
#         if (self.detectorName == "ORB"):
#             self.extractMessages=getOrbParameters()
#         self.cvb = CvBridge()
#         self.lROISub = rospy.Subscriber("/bumblebee/leftROI", Image, self.updatelROI)
#     def updatelROI(self,message):
#         latest=copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
#         ##create detector
#         print(self.settings)
#         d = cv2.ORB_create(self.settings.maxFeatures.data,
#                            self.settings.scale.data,
#                            self.settings.edge.data,
#                            self.settings.level.data,
#                            self.settings.wta.data,
#                            self.settings.score.data,
#                            self.settings.patch.data)
#         kp = d.detect(latest, None)
#         print(len(kp))
#         self.s.updateImage(cv2.drawKeypoints(latest, kp, None, color=(0, 255, 0), flags=0))
#         self.wait=False
#     def AnalyzeDataSet(self):
#         output=DataSetFeatures()
#         r=resetDatasetRequest()
#         self.resetSrv(r)
#         time.sleep(0.1)
#         self.settings = copy.deepcopy(self.extractMessages[0])
#         next=publishImageRequest()
#         next.command.data="Next"
#         self.publishSrv(next)
#         time.sleep(0.5)
#         index=0
#         for i in self.extractMessages:
#             self.settings=copy.deepcopy(i)
#             print(str(index)+"/"+str(len(self.extractMessages)))
#             next = publishImageRequest()
#             next.command.data = "Current"
#             self.wait=True
#             self.publishSrv(next)
#             while(self.wait):
#                 time.sleep(0.1)
#             index=index+1

# print("made stuff")
#         try:
class FeaturesAnalysis:
    def __init__(self,detectorName,display=True):
        try:
            rospy.wait_for_service("/front_end/setExtractorDetector", 4)
            print("Connected to /front_end/setExtractorDetector")
            self.setExtractor = rospy.ServiceProxy("/front_end/setExtractorDetector", setDetector)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            rospy.wait_for_service("/front_end/detectCurrent", 4)
            print("Connected to /front_end/detectCurrent")
            self.extractor = rospy.ServiceProxy("/front_end/detectCurrent", detectCurrent)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            rospy.wait_for_service("/dataset/publishImage", 4)
            print("Connected to /dataset/publishImage")
            self.publishSrv = rospy.ServiceProxy("/dataset/publishImage",publishImage)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            rospy.wait_for_service("/dataset/getDatasetInfo", 4)
            print("Connected to /dataset/getDatasetInfo")
            self.infoSrv = rospy.ServiceProxy("/dataset/getDatasetInfo",getDatasetInfo)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            rospy.wait_for_service("/dataset/resetDataset", 4)
            print("Connected to /dataset/resetDataset")
            self.resetSrv = rospy.ServiceProxy("/dataset/resetDataset", resetDataset)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        self.startTime=time.strftime('%d_%m_%H_%M_%S')
        self.display=display
        self.detectorName=detectorName
        self.index=0
        if(self.detectorName=="ORB"):
            self.extractMessages=getOrbParameters()
        self.cvb = CvBridge()
        self.lROISub = rospy.Subscriber("/bumblebee/leftROI", Image, self.updatelROI)
        self.a=None
        #if not os.path.exists(self.getFullDir()):
        #    os.makedirs(self.getFullDir())
    def updatelROI(self,message):
        self.a=copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
        print("newimage")
    def getFullDir(self):
        return self.outRootDirectory+"/"+self.dataSetName+"/"+self.startTime+"_"+self.detType
    def AnalyzeDataSet(self):
        output=DataSetFeatures()
        r=resetDatasetRequest()
        self.resetSrv(r)
        set = setDetectorRequest()
        set.Name.data = "ORB"
        set.orbConfig = self.extractMessages[0]
        self.setExtractor(set)
        time.sleep(0.1)
        next=publishImageRequest()
        next.command.data="Next"
        self.publishSrv(next)
        time.sleep(0.5)
        for i in self.extractMessages:
            print(i)
            next = publishImageRequest()
            next.command.data = "Current"
            self.publishSrv(next)
            time.sleep(0.5)
            set.orbConfig=self.extractMessages[0]
            set.detection=True
            self.setExtractor(set)
            time.sleep(0.1)
            a=detectCurrentRequest()
            result=self.extractor(a)
            print(result)
            time.sleep(3)
        return "a"
    #     if(self.detType=="FAST"):
    #         settings=getFastParameters()
    #         defaultMessage=extractFASTRequest()
    #     elif(self.detType=="SIFT"):
    #         settings=getSIFTParameters()
    #         defaultMessage=extractSIFTRequest()
    #     elif(self.detType=="ORB"):
    #         settings=getOrbParameters()
    #         defaultMessage=extractORBRequest()
    #     elif(self.detType=="SURF"):
    #         settings=getSURFParameters()
    #         defaultMessage=extractSURFRequest()
    #     elif(self.detType=="AKAZE"):
    #         settings=getAKAZEParameters()
    #         defaultMessage=extractAKAZERequest()
    #     elif(self.detType=="BRISK"):
    #         settings=getBRISKParameters()
    #         defaultMessage=extractBRISKRequest()
    #     endSeq = False
    #     ind=0
    #     while(not endSeq):
    #         ##analyze a single image
    #         currentDetector=0
    #         singleFrame=FrameFeatures()
    #         for detectorSettings in settings:
    #             outputImageName=self.outRootDirectory+"/tempExtractImage.png"
    #             print("["+str(currentDetector)+"/"+str(len(settings)-1)+"]"+self.dataNode.data.getStatusString() + "--" + self.dataNode.data.getCurrentDir())
    #             defaultMessage.config=detectorSettings
    #             defaultMessage.imageDir.data=self.dataNode.data.getCurrentDir()
    #             if(self.display):
    #                 defaultMessage.outputDir.data=outputImageName
    #             else:
    #                 defaultMessage.outputDir.data=""
    #             t=copy.deepcopy(defaultMessage)
    #             singleFrame.messages.append(t)##need a deep copy for some reason
    #             reply=self.extract(defaultMessage)
    #             if(self.display):
    #                 img = cv2.imread(outputImageName, cv2.IMREAD_COLOR)
    #                 cv2.imshow('Features', img)
    #                 cv2.waitKey(1)
    #             singleFrame.responses.append(reply)
    #             currentDetector += 1
    #         ###draw and save the images for the [min,avg,stdDev1,and max]
    #         statsIndexes=[singleFrame.getMinIndex(),singleFrame.getMedianIndex(),singleFrame.getMedianPlusSdIndex(),singleFrame.getMaxIndex()]
    #         #print(singleFrame.getAsList()[0])
    #         #print(statsIndexes)
    #         statsCount=0
    #         for i in statsIndexes:
    #             displayMessage=copy.deepcopy(singleFrame.messages[i])
    #             displayMessage.outputDir.data=outputImageName
    #             self.extract(displayMessage)
    #             img = cv2.imread(outputImageName, cv2.IMREAD_COLOR)
    #             imageFileName = self.getFullDir() + "/" + str(ind) + "_" + str(statsCount) + ".ppm"
    #             cv2.imwrite(imageFileName, img)
    #             singleFrame.imageDir.append(imageFileName)
    #             statsCount+=1
    #
    #         #a = a[a.rfind("/"):(len(a) - 4)]
    #         pickleName=str(ind)+"_data.p"
    #         pickle.dump(singleFrame, open(self.getFullDir() + "/"+pickleName, "wb"))
    #         output.frames.append(self.getFullDir() + "/"+pickleName)
    #         next=nextFrameRequest()
    #         next.Forward.data=True
    #         response=self.dataNode.updateFrame(next)
    #         if(response.Status.data=="End"):
    #             endSeq=True
    #         ind+=1
    #     return output

class FrameFeatures:
    def __init__(self):
        self.messages=[]
        self.responses=[]
        self.imageDir=[]
    def getSD(self,listindex):
        ##get standard deviation of left features
        if(len(self.getAsList()[listindex])>2):
            return statistics.stdev(self.getAsList()[listindex])
        else:
            return 0
    def getMean(self,listindex):
        return statistics.mean(self.getAsList()[listindex])
    def getMin(self,listindex):
        return min(self.getAsList()[listindex])
    def getMax(self,listindex):
        return max(self.getAsList()[listindex])
    def getAsList(self):
        listTimes=[]
        listnFeatures=[]
        for i in self.responses:
            listTimes.append(i.averageTime.data)
            listnFeatures.append(i.nleft.data)
        return [listnFeatures,listTimes]
    def getMaxIndex(self):
        return np.abs(np.array(self.getAsList()[0])-self.getMax(0)).argmin()
    def getMinIndex(self):
        return np.abs(np.array(self.getAsList()[0]) - self.getMin(0)).argmin()
    def getMedianIndex(self):
        return np.abs(np.array(self.getAsList()[0])-self.getMean(0)).argmin()
    def getMedianPlusSdIndex(self):
        return np.abs(np.array(self.getAsList()[0]) -(self.getMean(0)+ self.getSD(0))).argmin()




def drawProcVsNfeat(inDataSetFeatures):
    sty.use("seaborn")
    averageTime=[]
    nFeatures=[]
    for frame in inDataSetFeatures.frames:
        averageTime.append(frame.getMean(1))
        nFeatures.append(frame.getMean(0))
    plt.scatter(nFeatures,averageTime)
    plt.xlabel("Average Number of Features Extracted")
    plt.ylabel("Average processing time (ms)")
    plt.show()


class DataSetFeatures:
    def __init__(self):
        self.frames = []
    def drawGraph(self):
        sty.use("seaborn")
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        x = np.arange(0, len(self.frames), 1)
        # get left numbers
        leftMax = []
        leftMin = []
        leftTopDev = []
        leftMinDev = []
        leftMed = []
        timMax = []

        timMin = []
        timTopDev = []
        timMinDev = []
        timMed = []
        LeftListIndex = 0
        TimeListIndex = 1
        for i in self.frames:
            #unpickle object
            print(i)
            currentFrame=pickle.load(open(i,"rb"))

            leftMax.append(currentFrame.getMax(LeftListIndex))
            leftMin.append(currentFrame.getMin(LeftListIndex))
            leftMed.append(currentFrame.getMean(LeftListIndex))
            leftTopDev.append(currentFrame.getMean(LeftListIndex) + currentFrame.getSD(LeftListIndex))
            leftMinDev.append(currentFrame.getMean(LeftListIndex) - currentFrame.getSD(LeftListIndex))
            timMax.append(currentFrame.getMax(TimeListIndex))
            timMin.append(currentFrame.getMin(TimeListIndex))
            timMed.append(currentFrame.getMean(TimeListIndex))
            timTopDev.append(currentFrame.getMean(TimeListIndex) + currentFrame.getSD(TimeListIndex))
            timMinDev.append(currentFrame.getMean(TimeListIndex) - currentFrame.getSD(TimeListIndex))
        ax1.fill_between(x, leftMax, leftMin, color='black', alpha=0.5, linewidth=0.0)
        ax1.fill_between(x, leftTopDev, leftMinDev, color='darkred', alpha=0.2, linewidth=0.0)
        ax1.plot(x, leftMed, mew=0.1, marker='o', color='black')
        ax1.set_ylim(ymin=0.0)
        ax1.set_ylabel("Number of features")

        ax2.fill_between(x, timMax, timMin, color='black', alpha=0.2, linewidth=0.0)
        ax2.fill_between(x, timTopDev, timMinDev, color='darkblue', alpha=0.2, linewidth=0.0)
        ax2.plot(x, timMed, mew=0.1, marker='o', color='black')
        ax2.set_ylim(ymin=0.0)
        ax2.set_ylabel("Processing Time (ms)")
        plt.show()
        return [fig, (ax1, ax2)]
    def getBestDetectorsSettings(self):
        #define best detectors settings as Max,0.9Max, 0.8Max,0.7Max,stdDev,Mean
        max=[]
        max9=[]
        max8=[]
        max7=[]
        maxdev=[]
        maxmean=[]
        for frameIndex in self.frames:
            print("loading from ",frameIndex)
            currentFrame = pickle.load(open(frameIndex, "rb"))
            nfeatures=currentFrame.getAsList()[0]
            maximumFeaturesFound=currentFrame.getMax(0)
            stdDev_mean=currentFrame.getSD(0)+currentFrame.getMean(0)
            ##get the index numbers
            maxIndex = currentFrame.getMaxIndex()
            max9Index = np.abs(np.array(nfeatures) - 0.9*maximumFeaturesFound).argmin()
            max8Index= np.abs(np.array(nfeatures) - 0.8*maximumFeaturesFound).argmin()
            max7Index=np.abs(np.array(nfeatures) - 0.7*maximumFeaturesFound).argmin()
            maxdevIndex=currentFrame.getMedianPlusSdIndex()
            meanIndex=currentFrame.getMedianIndex()
            max.append([currentFrame.messages[maxIndex],currentFrame.responses[maxIndex]])
            max9.append([currentFrame.messages[max9Index],currentFrame.responses[max9Index]])
            max8.append([currentFrame.messages[max8Index],currentFrame.responses[max8Index]])
            max7.append([currentFrame.messages[max7Index],currentFrame.responses[max7Index]])
            maxdev.append([currentFrame.messages[maxdevIndex],currentFrame.responses[maxdevIndex]])
            maxmean.append([currentFrame.messages[meanIndex],currentFrame.responses[meanIndex]])
        return [max,max9,max8,max7,maxdev,maxmean]


