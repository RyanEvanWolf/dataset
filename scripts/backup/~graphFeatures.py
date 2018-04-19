#!/usr/bin/env python

import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as sty
from getFeatures import datasetFeatures,frameFeatures
defaultName="/media/ryan/Markov/DetectorOutput/D4/31_12_06-27-23-ORB/MetaData.p"


data=pickle.load(open(defaultName,"rb"))
bestSettings=data.getBestDetectorsSettings()

def getAsList(tupleData,columnIndex):
    outList=[]
    x=[]
    count=0
    for i in tupleData:
        x.append(count)
        if(columnIndex==2):
            ##is time, convert to milli seconds
            outList.append(i[columnIndex]*1000)
        else:
            outList.append(i[columnIndex])
        count=count+1
    return [x,outList]

##just for reference for the difference indexes
#[self.settings[maxIndex], currentFrame.nFound[maxIndex],currentFrame.avgTime[maxIndex],"maxFeatures"])
##return [max,max9,max8,max7,maxdev,maxmean]

sty.use("seaborn")
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
for i in range(6):
    ax1.plot(getAsList(bestSettings[i],1)[0], getAsList(bestSettings[i],1)[1], mew=0.1, marker='o', c=np.random.rand(3,))

ax1.set_ylim(ymin=0.0)
ax1.set_ylabel("Number of features")

for i in range(6):
    ax2.plot(getAsList(bestSettings[i],2)[0], getAsList(bestSettings[i],2)[1], mew=0.1, marker='o', c=np.random.rand(3,))

ax2.set_ylim(ymin=0.0)
ax2.set_ylabel("Processing Time (ms)")
plt.show()

##POSSIBLE code to read filename from tkinter window instead of manual entry

# import pickle
#
# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
#
#
#
# class mainApp(QWidget):
#     def __init__(self,parent=None):
#         super(mainApp, self).__init__(parent)
#         fname="/media/ryan/Markov/DetectorOutput/D4/30_11_23_25_27_ORB/Data.p"#self.getfileName()
#         print("loading data From "+str(fname))
#         data=pickle.load(open(fname,"rb"))
#         #data.drawGraph()
#         ans=data.getBestDetectorsSettings()
#         print("got answer")
#         pickle.dump(ans, open("/home/ryan/bestData.p", "wb"))
#         print("saved")
#         self.close()
#     def getfileName(self):
#         dlg = QFileDialog()
#         #dlg.setFileMode(QFileDialog.AnyFile)
#         ans=dlg.getOpenFileName(self,'OpenFile','/media/Markov/DetectorOutput','pickledObjects(.p)')
#         return ans[0]
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = mainApp()
#     sys.exit(app.exec_())