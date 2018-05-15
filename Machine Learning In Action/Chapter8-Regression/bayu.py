from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    fr = open(filename)
    xArr = []; yArr = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr.

def lwlr(testPoint, xArr, yArr, k=1.0):
