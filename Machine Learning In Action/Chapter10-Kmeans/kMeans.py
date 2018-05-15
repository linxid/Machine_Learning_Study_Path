import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    函数说明:载入数据,与其他并无不同
    :param fileName:
    :return:
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        length = len(curLine)
        # 将数据转换为浮点数类型
        for i in range(length):
            curLine[i] = float(curLine[i])
        dataMat.append(curLine)
    return dataMat

def distEclud(vecA, vecB):
    """
    函数说明:
        求解两个向量的欧式距离
    :param vecA: 向量A
    :param vecB: 向量B
    :return: 欧氏距离
    """
    # 计算欧氏距离,用到了多个numpy的函数,并不是math函数,主要因为对(数组)矩阵进行操作
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    """
    函数说明:
        产生随机的k个点,作为k个簇的质心,由随机数生成
    :param dataSet: 数据集
    :param k: 被划分成的簇数
    :return: 每个簇的质心
    """
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    # 建立k*n的矩阵来存储质心
    for j in range(n):# 遍历每个特征,对每个特征循环
        minJ = min(dataSet[:, j])   # 每一列的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)   # 每一列数据的范围,即最大值-最小值
        # 生成随机数矩阵,使用numpy的random函数,而不是random库
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, creatCent = randCent):
    """
    函数说明:
        整套的k均值算法,首先初始化质心,然后对每个点,进行簇划分,然后更新质心点

    :param dataSet: 待划分数据集
    :param k:划分成k个簇
    :param distMeas:求欧氏距离的函数
    :param creatCent:初始化质心函数
    :return:
        centroids:k个簇的质心
        clusterAssment:每个点所属的簇和欧氏距离

    """
    m = np.shape(dataSet)[0]    # 数据集的样本数
    clusterAssment = np.mat(np.zeros((m, 2)))   # 建立一个矩阵,存储每个样本所属的簇和欧氏距离
    centroids = creatCent(dataSet, k)   # 初始化k个簇的质心点
    clusterChanged = True   # 标志位,表示是否需要继续进行簇的更新
    while clusterChanged:   # 根据标志位进行循环
        clusterChanged = False
        for i in range(m):  # 对m个样本进行遍历,依次求每个点到每个簇质心的距离
            minDist = np.inf    # 设置最小距离
            minIndex = -1   # 设置所属簇的类别
            for j in range(k):  # 对k个簇进行遍历
                # 求样本点到每个质心的距离
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                # 寻找一个样本点到质心的最小距离和该质心所属的簇
                if distJI < minDist:
                    minDist = distJI    # 记录和更新最小值
                    minIndex = j    # 设置簇类别

            if clusterAssment[i,0] != minIndex:     # 如果簇类别依然被更新,则继续循环,知道簇不再更新
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2     # 存储类别和欧氏距离
        print(centroids)

        # 点划分后,对簇进行更新,
        for cent in range(k):
            # 求解数据集中属于某个簇的所有数据
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0],:]
            # 求这些数据的均值,对质心进行更新
            centroids[cent, :] = np.mean(ptsInClust, axis = 0)
    return centroids, clusterAssment

def plotData(dataMat, centroids, clusterAssing):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data1 = dataMat[np.nonzero(clustAssing[:, 0].A == 0)[0], :]
    data2 = dataMat[np.nonzero(clustAssing[:, 0].A == 1)[0], :]
    data3 = dataMat[np.nonzero(clustAssing[:, 0].A == 2)[0], :]
    data4 = dataMat[np.nonzero(clustAssing[:, 0].A == 3)[0], :]
    xMat1 = data1[:, 0].tolist()
    yMat1 = data1[:, 1].tolist()
    xMat2 = data2[:, 0].tolist()
    yMat2 = data2[:, 1].tolist()
    xMat3 = data3[:, 0].tolist()
    yMat3 = data3[:, 1].tolist()
    xMat4 = data4[:, 0].tolist()
    yMat4 = data4[:, 1].tolist()

    ax.scatter(xMat1, yMat1,color='red')
    ax.scatter(xMat2, yMat2, color='blue')
    ax.scatter(xMat3, yMat3, color='green')
    ax.scatter(xMat4, yMat4, color='orange')

    xCentroids = centroids[:, 0].tolist()
    yCentroids = centroids[:, 1].tolist()
    ax.scatter(xCentroids, yCentroids, marker='+', s=80, color = 'black')

    plt.show()



if __name__ == '__main__':
    fileName = 'testSet2.txt'
    dataMat = np.mat(loadDataSet(fileName))
    myCentroids, clustAssing = kMeans(dataMat, 3)
    plotData(dataMat, myCentroids, clustAssing)









