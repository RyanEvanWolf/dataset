import os
import rosbag
import yaml
import cv2
import copy
BAG_FOLDER="Bags"
FEATURES_FOLDER="Features"

UNSORTED_FEATURES_PICKLE="baseDetection.p"
OPERATING_CURVES_PICKLE="OperatingCurves.p"
ALGORITHM_STATISTICS="AlgorithmOneStats.p"

detectorTableDirectory="/media/ryan/EXTRA/output/detectorLookupTable.yaml"

def OperatingCurveIDs():
    Ids=["Maximum","0.9Maximum",
        "0.8Maximum","+Deviation",
        "Mean","-Deviation","Minimum"]
    return Ids

def OperatingCurveColours():
    col=[(1,0.3,0,0.8),(1,0.5,0,0.8),(1,0.7,0,0.8),(0,0.5,0.2,0.8),(0,0.5,0.6,0.8),(0,0.5,0.8,0.8),(0,0,0,0.8)]
    return col

def getBagID(bagFile):
    outID=bagFile[bagFile.find("_")+1:bagFile.rfind(".bag")]
    return outID

def getDetectorTable(detectorTableDirectory="/media/ryan/EXTRA/output/detectorLookupTable.yaml"):
    detTableSource=open(detectorTableDirectory, 'r') 
    detTable=yaml.load(detTableSource)
    detTableSource.close()
    return detTable

def unpackBag(bagFile,maxImages=-1,verbose=False):
    Output={}
    Output["Left"]=[]
    Output["Right"]=[]
    if(verbose):
        print("extracting rosbag data from "+bagFile)
    inputBag=rosbag.Bag(bagFile)
    for topic,msg,t in inputBag.read_messages(topics=['/bumblebee/left/ROI','/bumblebee/right/ROI']):
        if(topic=="/bumblebee/left/ROI"):
            Output["Left"].append(msg)
        if(topic=="/bumblebee/right/ROI"):
            Output["Right"].append(msg)
        if(verbose):
            print(len(Output["Left"]))
        if((maxImages!=-1)and((len(Output["Left"])+len(Output["Right"]))>=2*maxImages)):
            if(verbose):
                print("halted image loading at total = "+str(maxImages))
            break
    inputBag.close()  
    return Output
def getDefaultDirectories():
    Direct={}
    Direct["Root"]="/home/ryan"
    Direct["DataSet"]="DATA3"
    Direct["BagFolder"]="Bags"
    Direct["RootOut"]="/media/ryan/EXTRA/output"
    Direct["FeaturesFolder"]="Features"
    Direct["SimulationFolder"]="Simulation"
    Direct["CurvePickle"]="operatingCurves.p"
    return Direct


def getImageSet(metaDataInfo,loopNumber):
    rootDir=metaDataInfo[:metaDataInfo.rfind("/")]
    f=open(metaDataInfo,'r')
    loopData=f.readlines()
    loopName,startImage,endImage=loopData[loopNumber-1][:-1].split(",")
    startImage=int(startImage[:startImage.rfind(".")])
    endImage=int(endImage[:endImage.rfind(".")])
    setImages=[str(item).zfill(5)+".ppm" for item in range(startImage,endImage+1)]
    limages=[rootDir+"/left/rectified/"+item for item in setImages]
    rimages=[rootDir+"/right/rectified/"+item for item in setImages]
    f.close()
    return rootDir,limages,rimages

def loadImages(imageList):
    return copy.deepcopy([cv2.imread(item,cv2.IMREAD_GRAYSCALE) for item in imageList])

class Directories:
    def __init__(self,foldStruct=getDefaultDirectories()):
        self.Direct=foldStruct
        # self.baseDirectory =baseDirectory
        # self.bagDir=self.baseDirectory+"/"+BAG_FOLDER
        # if(outputDirectory!=""):
        #     self.featDir=outputDirectory+"/"+FEATURES_FOLDER
        # else:
        #     self.featDir=self.baseDirectory+ "/"+ FEATURES_FOLDER
    def getBagPath(self):
         return (self.Direct["Root"]+"/"+self.Direct["DataSet"]
                +"/"+self.Direct["BagFolder"])
    def getBags(self):
        return os.listdir(self.getBagPath())
    def getBagName(self,ID):
        return self.getBagPath()+"/stereo_"+ID+".bag"
    def getCurvePickle(self,loopID,detectorType):
        return self.getFeaturePath(loopID,detectorType)+".p"
    def getFeaturePath(self,loopID,detectorType):
        outPath=(self.Direct["RootOut"]+"/"+self.Direct["FeaturesFolder"]
                +"/"+self.Direct["DataSet"]+"/"+loopID+
                "/"+detectorType)
        return outPath
    def getAlgorithmStatsPickle(self,loopID):
        return self.getFeaturePath(loopID)+"/"+ALGORITHM_STATISTICS
