#!/usr/bin/env python
from bumbleDataSet import bumbleDataSet,bumbleDataSetNode

##Ros specific imports
import rospy
import tf
from std_msgs.msg import Int32, Bool, Int8, Float64,String

#system imports
from threading import Thread
import datetime
import numpy as np
import sys
import time
from Queue import Queue

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTabWidget, QFormLayout, QLineEdit,QCheckBox, QComboBox
from PyQt5.QtCore import pyqtSlot, QThread, QObject, pyqtSignal, QMutex, QSettings
from PyQt5.QtGui import QKeyEvent
import cv2

from dataset.srv import loadDataSet,loadDataSetRequest,loadDataSetResponse
from dataset.srv import nextFrame, nextFrameResponse, nextFrameRequest

from front_end.srv import setDetector,setDetectorResponse,setDetectorRequest
from front_end.msg import ORB,StereoFrame



from drawingFunctions import stereoDrawing


class mainApp(QWidget):
    def __init__(self,parent=None):
        super(mainApp, self).__init__(parent)
        self.playbtn = QPushButton("Playback", self)
        self.prevbtn = QPushButton("Prev", self)
        self.nextbtn = QPushButton("Next", self)
        self.savebtn = QPushButton("Save", self)
        self.newbtn = QPushButton("newDataSet", self)

        self.playbtn.clicked.connect(self.play)
        self.prevbtn.clicked.connect(self.prev)
        self.nextbtn.clicked.connect(self.nxt)
        self.savebtn.clicked.connect(self.save)
        self.newbtn.clicked.connect(self.newData)
        self.worker=workerClass()
        self.worker.daemon=True

        self.node = rospy.init_node('dataset_node')
        self.initUI()
        self.show()
        self.worker.start()
    def initUI(self):
        l=QFormLayout(self)
        l.addRow(self.playbtn)
        l.addRow(self.prevbtn)
        l.addRow(self.nextbtn)
        l.addRow(self.savebtn)
        l.addRow(self.newbtn)
        self.setLayout(l)
    def play(self):
        self.worker.commands.put("playback")
    def prev(self):
        self.worker.commands.put("previous")
    def nxt(self):
        self.worker.commands.put("next")
    def save(self):
        self.worker.commands.put("save")
    def newData(self):
        self.worker.settingsUpdate.put("XML File from edit box")
    def closeEvent(self,QCloseEvent):
        self.worker.settingsUpdate.put("xxx")


class workerClass(Thread):
    def __init__(self):
        super(workerClass,self).__init__()
        self.settingsUpdate=Queue()
        self.commands=Queue()
        self.data = bumbleDataSet()
    def run(self):
        alive=True
        while(alive):
            if(self.settingsUpdate.qsize()>0):
                alive=self.updateSettings(self.settingsUpdate.get())
            if(alive):
                if(self.commands.qsize()>0):
                    self.commandResponse(self.commands.get())
    def updateSettings(self,message):
        print(message)
        if(message=="exit"):
            return False
        else:
            return True
    def commandResponse(self,message):
        if(message=="playback"):
            print("send")
        elif(message=="previous"):
            print("check prev")
        elif(message=="next"):
            print("next styff")
        elif(message=="save"):
            print("save my data")
        elif(message==""):
            print("aaaa")
        else:
            print("invalid command received")

global receivedData
global count
global feedback
def callback(data):
    global receivedData
    global count
    global feedback
    count= count+1
    print("recived packet NUMBER",count)
    feedback.drawMatches(data.matches)
    receivedData=True

if __name__ == '__main__':
    rospy.init_node('DataSet')
    receivedData = False
    count = 0
    feedback = stereoDrawing("bumblebee/leftROI", "bumblebee/rightROI", "bumblebee/leftRectified",
                             "bumblebee/rightRectified", "front_end/ROIleft", "front_end/ROIright")
    #app = QApplication(sys.argv)
    #main = mainApp()
    ##main.worker.data.loadFiles(sys.argv[1])
    #app.exec_()
    dataNode = bumbleDataSetNode()
    rospy.wait_for_service("dataset/loadDataSet", 4)
    rospy.wait_for_service("front_end/setDetector",4)
    dataRootSrv=rospy.ServiceProxy("dataset/loadDataSet",loadDataSet)
    detSrv=rospy.ServiceProxy("front_end/setDetector",setDetector)

    stereoSub=rospy.Subscriber("front_end/stereo",StereoFrame,callback)

    newRoot=loadDataSetRequest()
    newRoot.dataSetRootPath.data=str(sys.argv[1])
    dataRootSrv(newRoot)

    time.sleep(0.2)  ##waitfor server to setup
    print("loaded Dataset "+ str(sys.argv[1]))
    m = setDetectorRequest()
    settings=ORB()
    settings.maxFeatures.data=10000
    settings.scale.data = 1.5
    settings.edge.data = 4
    settings.level.data =3
    settings.wta.data = 2
    settings.score.data = cv2.ORB_HARRIS_SCORE
    settings.patch.data = 16
    m.orbConfig = settings
    m.Name.data="ORB"
    m.detection=True
    detSrv(m)
    m.detection=False
    detSrv(m)
    time.sleep(1)
    next = nextFrameRequest()
    next.Forward.data = True
    print(dataNode.data.getStatusString())
    receivedData=False
    dataNode.updateFrame(next)
    #while(not receivedData):
    ##    time.sleep(0.5)
     #   print("waiting-1")
    #print(dataNode.data.getStatusString())
    #receivedData=False
    #dataNode.updateFrame(next)
    #while(not receivedData):
    #    time.sleep(0.5)
    #    print("waiting-2")
    print("completed")
    rospy.spin()