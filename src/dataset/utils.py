import os
import rosbag
import yaml
BAG_FOLDER="Bags"
FEATURES_FOLDER="Features"

UNSORTED_FEATURES_PICKLE="baseDetection.p"
OPERATING_CURVES_PICKLE="OperatingCurves.p"
ALGORITHM_STATISTICS="AlgorithmOneStats.p"

detectorTableDirectory="/media/ryan/EXTRA/output/detectorLookupTable.yaml"

def OperatingCurveIDs():
    Ids=["Maximum","0.9Maximum","0.8Maximum",
        "0.7Maximum","0.6Maximum","+Deviation",
        "Mean","-Deviation","Minimum"]
    return Ids


def getBagID(bagFile):
    outID=bagFile[bagFile.find("_")+1:bagFile.rfind(".bag")]
    return outID

def getDetectorTable():
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
    def getCurvePickle(self,loopID):
        return self.getFeaturePath(loopID)+"/"+OPERATING_CURVES_PICKLE
    def getFeaturePath(self,loopID,detectorType):
        outPath=(self.Direct["RootOut"]+"/"+self.Direct["FeaturesFolder"]
                +"/"+self.Direct["DataSet"]+"/"+loopID+
                "/"+detectorType)
        return outPath
    def getAlgorithmStatsPickle(self,loopID):
        return self.getFeaturePath(loopID)+"/"+ALGORITHM_STATISTICS
