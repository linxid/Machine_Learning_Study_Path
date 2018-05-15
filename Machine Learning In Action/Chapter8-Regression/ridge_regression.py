from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    """
    函数说明：
        载入数据
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

def ridgeRegres(xMat, yMat, lam=0.2):
    """
    函数说明：
        岭回归的算法实现
    :param xMat:训练数据样本
    :param yMat:训练数据标签
    :param lam:lambda参数
    :return:
        ws：返回的weight参数
    """
    xTx = xMat.T*xMat
    # 在原有算法的基础上，增加lambda修正项，避免不可逆
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('奇异矩阵，不能求逆')
        return
    # print(np.shape(xMat), np.shape(yMat), np.shape(denom))
    # 一定注意矩阵的维数，保证可以相乘
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    """
    函数说明：
        测试岭回归算法，使用
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr);yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis=0)
    yMat = yMat = yMean
    xMeans = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def plotwMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    ax_title_text = ax.set_title(u'log(lambda)与回归系数的关系',FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'log(lanbda)',FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数',FontProperties=font)

    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    plotwMat()