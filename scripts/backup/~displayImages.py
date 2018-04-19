#!/usr/bin/env python
from sensor_msgs.msg import Image
import sys
import rospy
import cv2
from cv_bridge import CvBridge
import copy
import numpy as np
from front_end.msg import StereoFrame,FrameTracks,Window
from std_msgs.msg import Bool
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel,QMainWindow
from PyQt5.QtGui import QPixmap, QPainter, QImage


class imageWindow(QWidget):
    def __init__(self, imgSub, parent=None):
        QWidget.__init__(self, parent=parent)
        self.imagesRef=imgSub
        self.resize(500,500)
        self.frame = imageFrame()
        lay = QVBoxLayout(self)
        lay.addWidget(self.frame)
        self.show()
    def redraw(self,q_image):
        self.frame.p=QPixmap(q_image)
        self.update()

class singleImageViewer(QWidget):
    def __init__(self,imgTopic,colourImage=False,parent=None):
        QWidget.__init__(self,parent=parent)
        self.resize(500,500)
        self.frame=imageFrame()
        lay=QVBoxLayout(self)
        lay.addWidget(self.frame)
        self.show()
        self.cvb = CvBridge()
        self.topic = imgTopic
        self.lroSub = rospy.Subscriber(self.topic, Image, self.updateImage)
        self.colour=colourImage
    def redraw(self,q_image):
        self.frame.p=QPixmap(q_image)
        self.update()
    def updateImage(self,message):
         newImage=self.cvb.imgmsg_to_cv2(message)
         if(not self.colour):
             newImage=cv2.cvtColor(newImage,cv2.COLOR_GRAY2RGB)
         qImg = QImage(newImage.data, newImage.shape[1],newImage.shape[0], 3 *newImage.shape[1],
                          QImage.Format_BGR888)
         self.redraw(QPixmap(qImg))

class TrackingViewer(imageWindow):
    def __init__(self, imageSub):
        imageWindow.__init__(self, imageSub)
        self.onPubDraw = rospy.Subscriber("front_end_window/State", Window, self.feedback)
        self.setWindowTitle("front_end_window/State")
        self.updatedPub = rospy.Publisher("front_end_window/State/updated", Bool, queue_size=5)
    def feedback(self,message):
        print(message.frames[0].inliers[0].distance)
        if(len(message.tracks)>=1):
            drawnImage = copy.deepcopy(self.imagesRef.limg)
            drawnImage = cv2.cvtColor(drawnImage,cv2.COLOR_GRAY2RGB)
            latestInterFrame=message.tracks[-1]
            print(len(latestInterFrame.currentInlierIndexes))
            inl=0
            outl=0
            for n in range(len(latestInterFrame.currentInlierIndexes)):
                px=int(message.frames[-1].inliers[latestInterFrame.currentInlierIndexes[n]].stereo.leftFeature.imageCoord.x)
                py=int(message.frames[-1].inliers[latestInterFrame.currentInlierIndexes[n]].stereo.leftFeature.imageCoord.y)
                prex=int(message.frames[-2].inliers[latestInterFrame.previousInlierIndexes[n]].stereo.leftFeature.imageCoord.x)
                prey=int(message.frames[-2].inliers[latestInterFrame.previousInlierIndexes[n]].stereo.leftFeature.imageCoord.y)
                cv2.circle(drawnImage, (px, py), 2, (0, 200, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                cv2.circle(drawnImage, (prex, prey), 2, (0, 20, 150), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                cv2.line(drawnImage, (prex, prey), (px, py), (0,100,100), 1)
                if(latestInterFrame.motionInlierMask[n]):
                    cv2.circle(drawnImage, (px, py), 2, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                    cv2.circle(drawnImage, (prex, prey), 2, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                    cv2.line(drawnImage, (prex, prey), (px, py), (255, 0, 0), 1)
                    outl=outl+1
                else:
                    inl=inl+1

            qImg = QImage(drawnImage.data, drawnImage.shape[1],drawnImage.shape[0], 3 *drawnImage.shape[1],
                          QImage.Format_RGB888)
            self.redraw(QPixmap(qImg))
            print(message.tracks[-1].R)
            print(message.tracks[-1].T)
            print("InlierPercentage",100*inl/float(inl+outl))
            #cv2.imshow("tracks",drawnImage)
        msg=Bool()
        self.updatedPub.publish(msg)
class imageFrame(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.p = None
        self.show()
    def paintEvent(self, event):
        if self.p:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(self.rect(), self.p)

class StereoViewer(imageWindow):
    def __init__(self,imageSub):
        imageWindow.__init__(self,imageSub)
        self.onPubDraw = rospy.Subscriber("/front_end/stereo", StereoFrame, self.feedback)
        self.setWindowTitle("/front_end/stereo")
        self.updatedPub = rospy.Publisher("/front_end/stereo/updated", Bool, queue_size=5)
    def feedback(self,message):
        buff=copy.deepcopy(self.imagesRef.limg)
        left=cv2.cvtColor(self.imagesRef.limg,cv2.COLOR_GRAY2RGB)

        left[:,:,0]=0.9*left[:,:,0]
        left[:,:,1]=0.05*left[:,:,1]
        left[:,:,2]=0.05*left[:,:,2]
        buff = copy.deepcopy(self.imagesRef.rimg)
        right=cv2.cvtColor(self.imagesRef.rimg,cv2.COLOR_GRAY2RGB)
        right[:,:,1]=0.3*right[:,:,1]
        right[:,:,2]=0.6*right[:,:,2]
        right[:,:,0]=0.1*right[:,:,0]
        combinedImage=cv2.addWeighted(left,0.5,right,0.5,25)
        backup=copy.deepcopy(combinedImage)

        for index in message.matches:
            rightPoint_x = index.rightFeature.imageCoord.x# + 1024  ##offset is size of left image width
            ##draw onto image
            cv2.circle(combinedImage, (int(rightPoint_x), int(index.rightFeature.imageCoord.y)), 2, (0, 200, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.circle(combinedImage, (int(index.leftFeature.imageCoord.x), int(index.leftFeature.imageCoord.y)), 2,
                       (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.line(combinedImage, (int(index.leftFeature.imageCoord.x), int(index.leftFeature.imageCoord.y)),
                     (int(rightPoint_x), int(index.rightFeature.imageCoord.y)),
                     (0, 255, 0), 1)
        combinedImage = cv2.addWeighted(combinedImage, 0.1, backup, 0.9, 0)
        qImg = QImage(combinedImage.data, combinedImage.shape[1], combinedImage.shape[0], 3 * combinedImage.shape[1],
                      QImage.Format_RGB888)
        self.redraw(QPixmap(qImg))
        #cv2.imshow("oasofsaoasf",combinedImage)
        msg=Bool()
        self.updatedPub.publish(msg)

class imageSubscriber():
    def __init__(self):
        self.cvb = CvBridge()
        self.Lroitopic = "/bumblebee/leftROI"
        self.Rroitopic = "/bumblebee/rightROI"
        self.Ltopic = "/bumblebee/leftRectified"
        self.Rtopic = "/bumblebee/rightRectified"
        self.Rtopic = "/bumblebee/rightRectified"
        self.lroSub = rospy.Subscriber(self.Lroitopic, Image, self.updatelr)
        self.rroSub = rospy.Subscriber(self.Rroitopic, Image, self.updaterr)
        self.lSub = rospy.Subscriber(self.Ltopic, Image, self.updatel)
        self.rSub = rospy.Subscriber(self.Rtopic, Image, self.updater)
        self.limg = None
        self.rimg = None
        self.lroiimg = None
        self.rroiimg = None
    def updatel(self, message):
        self.limg = copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updater(self, message):
        self.rimg = copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updatelr(self, message):
        self.lroiimg = copy.deepcopy(self.cvb.imgmsg_to_cv2(message))
    def updaterr(self, message):
        self.rroiimg = copy.deepcopy(self.cvb.imgmsg_to_cv2(message))


if __name__ == '__main__':
    print("main")
    rospy.init_node('display')
    rospy.set_param("displays","LeftROI RightColour")
    listOfWindows=rospy.get_param("displays").split(" ")
    print(listOfWindows)
    print(sys.argv)
    #rospy.set_param("waitTopicName","/front_end/stereo/drawn")
    app = QApplication(sys.argv)
    a=imageSubscriber()
    imageWindows=[]
    for i in listOfWindows:
         if(i=="LeftROI"):
             imageWindows.append(singleImageViewer("/bumblebee/leftROI",True))
    #     if(i=="RightColour"):
    #         imageWindows.append(singleImageViewer("/bumblebee/colour/rightROI",True))

    #b=imageWindow()
    #l=StereoViewer(a)
    #r=TrackingViewer(a)
    #l222 = StereoViewer("/front_end/stereo", StereoFrame, a)
    #b=singleImageViewer()
    #leftFeat=drawFeatures("/bumblebee/leftRectified","/front_end/stereo")

    #leftImage= SingleImageDisplay("/bumblebee/leftROI")
    #leftImage.show()
    #stereoMatches=StereoDisplay("/bumblebee/leftRectified","/bumblebee/rightRectified","/front_end/stereo")
    #stereoMatches.show()
    #track=TrackingDisplay("bumblebee/leftRectified","/front_end/stereo","front_end_window/FrameTracksleft")
    #track.show()
    #track2=TrackingDisplay("bumblebee/rightRectified","/front_end/stereo","front_end_window/FrameTracksright",False)
    #track2.show()
    #track3=TrackingDisplay("bumblebee/leftRectified","/front_end/stereo","front_end_window/FrameTracks")
    #track3.show()
    sys.exit(app.exec_())


