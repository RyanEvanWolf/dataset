import os
import rosbag
BAG_FOLDER="Bags"
FEATURES_FOLDER="Features"

UNSORTED_FEATURES_PICKLE="baseDetection.p"
OperatingCurvesName="OperatingCurves.p"

def OperatingCurveIDs():
    Ids=["Maximum","0.9Maximum","0.8Maximum",
        "0.7Maximum","0.6Maximum","+Deviation",
        "Mean","-Deviation","Minimum"]
    return Ids


def getBagID(bagFile):
    outID=bagFile[bagFile.find("_")+1:bagFile.rfind(".bag")]
    return outID

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
class Directories:
    def __init__(self,baseDirectory,outputDirectory=""):
        self.baseDirectory =baseDirectory
        self.bagDir=self.baseDirectory+"/"+BAG_FOLDER
        if(outputDirectory!=""):
            self.featDir=outputDirectory+"/"+FEATURES_FOLDER
        else:
            self.featDir=self.baseDirectory+ "/"+ FEATURES_FOLDER
    def getBags(self):
        return os.listdir(self.bagDir)
    def getBagPath(self,ID):
        return self.bagDir+"/stereo_"+ID+".bag"
    def getCurvePickle(self,loopID,DetectorID):
        return self.getFeaturePath(loopID,DetectorID)+"/"+OperatingCurvesName
    def getFeaturePath(self,loopID):
        return self.featDir+"/"+loopID
    def getFeaturePickle(self,loopID):
        return self.getFeaturePath(loopID)+"/"+UNSORTED_FEATURES_PICKLE

