import os
BAG_FOLDER="Bags"
FEATURES_FOLDER="Features"

initialFeaturesName="baseDetection.p"
OperatingCurvesName="OperatingCurves.p"

def OperatingCurveIDs():
    Ids=["Maximum","0.9Maximum","0.8Maximum",
        "0.7Maximum","0.6Maximum","+Deviation",
        "Mean","-Deviation","Minimum"]
    return Ids


def getBagID(bagFile):
    outID=bagFile[bagFile.find("_")+1:bagFile.rfind(".bag")]
    return outID




class Directories:
    def __init__(self,baseDirectory):
        self.baseDirectory =baseDirectory
        self.bagDir=self.baseDirectory+"/"+BAG_FOLDER
        self.featDir=self.baseDirectory+ "/"+ FEATURES_FOLDER
    def getBags(self):
        return os.listdir(self.bagDir)
    def getBagPath(self,ID):
        return self.bagDir+"/stereo_"+ID+".bag"
    def getCurvePickle(self,loopID,DetectorID):
        return self.getFeaturePath(loopID,DetectorID)+"/"+OperatingCurvesName
    def getFeaturePath(self,loopID,DetectorID):
        return self.featDir+"/"+loopID
    def getFeaturePickle(self,loopID,DetectorID):
        return self.getFeaturePath(loopID,DetectorID)+"/"+initialFeaturesName

