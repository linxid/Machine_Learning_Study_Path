from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet():
    dataMat = [];  labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, c='red',s=20, marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, c='green', s=20, alpha=.5)
    x = np.arange(-3,3,0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha*dataMatrix.transpose()*error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(i+j+1.0) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            weights_array = np.append(weights_array, weights, axis=0)
            del(dataIndex[randIndex])

    weights_array = weights_array.reshape(numIter*m, n)
    return weights, weights_array

def plotWeights(weights_array1,weights_array2):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)
    fig, axs = plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=(20,10))

    x1 = np.arange(0,len(weights_array1), 1)
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text1 = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数的关系',FontProperties = font)
    axs0_ylabel_text1 = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text1,size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text1,size=20,weight='bold', color='black')

    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text1 = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text1,size=20, weight='bold', color='black')

    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_ylabel_text1 = axs[2][0].set_ylabel(u'W2',FontProperties=font)
    axs2_xlabel_text1 = axs[2][0].set_xlabel(u'迭代次數',FontProperties=font)
    plt.setp(axs2_xlabel_text1, weight='bold', size=20, color='black')
    plt.setp(axs2_ylabel_text1, weight='bold', size=20, color='black')

    x2 = np.arange(0,len(weights_array2), 1)
    axs[0][1].plot(x2,weights_array2[:, 0])
    axs0_title_text2 = axs[0][1].set_title(u'改进的随机梯度算法：回归系数和迭代次数的关系', FontProperties=font)
    axs0_ylabel_text2 = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text2, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text2, size=20, weight='bold', color='black')

    axs[1][1].plot(x2, weights_array2[:,1])
    axs1_ylabel_text2 = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text2, size=20, weight='bold', color='black')

    axs[2][1].plot(x2, weights_array2[:,2])
    axs2_xlabel_text2 = axs[2][1].set_xlabel(u'迭代次數', FontProperties=font)
    axs2_ylabel_text2 = axs[2][1].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text2, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text2, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat),labelMat)
    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)