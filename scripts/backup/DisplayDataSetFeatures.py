#!/usr/bin/env python

import pickle

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog



class mainApp(QWidget):
    def __init__(self,parent=None):
        super(mainApp, self).__init__(parent)
        fname="/media/ryan/Markov/DetectorOutput/D4/30_11_23_25_27_ORB/Data.p"#self.getfileName()
        print("loading data From "+str(fname))
        data=pickle.load(open(fname,"rb"))
        #data.drawGraph()
        ans=data.getBestDetectorsSettings()
        print("got answer")
        pickle.dump(ans, open("/home/ryan/bestData.p", "wb"))
        print("saved")
        self.close()
    def getfileName(self):
        dlg = QFileDialog()
        #dlg.setFileMode(QFileDialog.AnyFile)
        ans=dlg.getOpenFileName(self,'OpenFile','/media/Markov/DetectorOutput','pickledObjects(.p)')
        return ans[0]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = mainApp()
    sys.exit(app.exec_())


