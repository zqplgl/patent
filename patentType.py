#coding=utf-8
import os
import shutil
import cv2
import numpy as np
import math
import caffe
import pickle
import time
from objTypeClassifier import ObjTypeClassifier

class IPatentTypeClassifier():
    def __init__(self,modelDir,gpu_id):
        modelDir += '/' if not modelDir.endswith('/') else modelDir

        assert(os.path.exists(modelDir+'patentClass'))
        prototxt = modelDir+"patentClass/patentimageclass.prototxt"
        assert(prototxt)
        weightfile = modelDir+'patentClass/bvlc_googlenet_iter_14000.caffemodel'
        assert(weightfile)
        meanfile = modelDir+'patentClass/patentimageclassmean.binaryproto'
        assert(meanfile)
        labelfile = modelDir+'patentClass/patentAllTypeID4.txt'
        assert(labelfile)

        self.__gpu_id = gpu_id
        self.__classifier = ObjTypeClassifier(prototxt,weightfile,meanfile,gpu_id)
        self.__labelMap = [line.strip() for line in open(labelfile).readlines()]

    def classify(self,im):
        result = {}

        info = self.__classifier.classify(im,'loss2/fc')
        result['category'] = self.__labelMap[info[0]]
        result['score'] = info[1]

        return result

def run():
    modelDir = r'/home/zqp/gitlab/models'
    classifier = IPatentTypeClassifier(modelDir,0)

    picDir = r'/home/zqp/testpic/patentClass/'
    for picName in os.listdir(picDir):
        im = cv2.imread(picDir+picName)
        result = classifier.classify(im)
        print (result['category'],"****************",result['score'])
        cv2.imshow('im',im)
        if cv2.waitKey(0)==27:
            break

if __name__ == '__main__':
    run()

