import numpy as np
import matplotlib.pyplot as plt

def loadDataSet1(filename):
    """
    函数说明:
        导入数据,注意数据类型的变换
    :param filename:
    :return:
    """
    fr = open(filename)
    stringArr = [line.strip().split('\t') for line in fr.readlines()]
    dataArr = []
    for i in range(len(stringArr)):
        currLine = []
        for j in range(len(stringArr[0])):

            currLine.append(float(stringArr[i][j]))
        dataArr.append(currLine)
    return np.mat(dataArr)

def loadDataSet2(filename):
    """
    函数说明:
        导入数据,注意数据类型的变换
    :param filename:
    :return:
    """
    fr = open(filename)
    stringArr = [line.strip().split(' ') for line in fr.readlines()]
    dataArr = []
    for i in range(len(stringArr)):
        currLine = []
        for j in range(len(stringArr[0])):

            currLine.append(float(stringArr[i][j]))
        dataArr.append(currLine)
    return np.mat(dataArr)


def pca(dataMat, topNfeat=99999):
    """
    函数说明:
        PCA算法,对数据进行降维
    :param dataMat:
    :param topNfeat:
    :return:
    """
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDataMat = meanRemoved * redEigVects
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat, reconMat

def replaceNanWitMean():
    dataMat = loadDataSet2('secom.data')
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
        dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal

    return dataMat

# if __name__ == '__main__':
#     dataMat = loadDataSet1('testSet.txt')
#     lowDataMat, reconMat = pca(dataMat, 1)
#     m, n = np.shape(lowDataMat)
#     print(m, n)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], s=40, color='blue')
#     ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='+', s=50, color='red')
#     plt.show()

if __name__ == '__main__':
    dataMat = replaceNanWitMean()
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print(eigVals)

