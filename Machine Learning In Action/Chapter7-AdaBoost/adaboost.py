import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    """
    函数说明:
        建立数据集,包括样本数据和标签数据
    :return:
        dataMat:样本数据
        classLabels:分类标签数据
    """
    # 转换城矩阵形式来处理
    dataMat = np.mat([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])

    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels
def plotData(dataMat, classLabels):
    """
    函数说明:
        对上述数据点进行绘制
    :param dataMat:
    :param classLabels:
    :return:
    """
    dataMat1 = dataMat[np.nonzero(np.mat(classLabels)[:].A == 1.0)[1], :]
    dataMat2 = dataMat[np.nonzero(np.mat(classLabels)[:].A == -1.0)[1], :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(dataMat2)
    print(dataMat1)
    ax.scatter(dataMat1[:, 0].tolist(), dataMat1[:, 1].tolist(), marker='d', color='red')
    ax.scatter(dataMat2[:, 0].tolist(), dataMat2[:, 1].tolist(), marker='o', color='blue')

    plt.show()


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    函数说明:
        通过阈值比较对数据进行分类
    :param dataMatrix: 训练数据
    :param dimen: 特征维度,第几个特征
    :param threshVal: 阈值,
    :param threshIneq:不等号选择的标志位,是大于还是小于等于
    :return:
        retArray:对数据分类后的结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    # 返回矩阵初始化为1,不满足不等式的元素设为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] =  -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    函数说明:
        建立基分类器
        求数据集上最优的单层决策树
    :param dataArr: 样本数据
    :param classLabels: 标签数据
    :param D: 数据的权重向量
    :return:
    """

    dataMatrix = np.mat(dataArr);labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    # 迭代的次数
    numSteps = 10.0
    # 使用字典,存储最佳单层决策树的相关信息
    bestStump = {}
    # 存储最佳预测分类结果
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # 对所有特征进行遍历
    for i in range(n):
        # 根据迭代次数确定步长,求当前i特征的最大和最小值
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        stepSize  = (rangeMax - rangeMin)/numSteps
        #
        for j in range(-1, int(numSteps) + 1):
            # 不确定哪个属于类-1或者1
            for inequal in ['lt', 'gt']:
                # 每个划分的阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 使用单层决策树进行分类,得到每个样本的分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 误差矩阵,存储每个样本的是否预测准确
                errorArr = np.mat(np.ones((m, 1)))
                # 预测准确的令为0
                errorArr[predictedVals == labelMat] = 0
                # 总误差,对每个样本的误差进行加权
                weightedError = D.T*errorArr    # 每个样本的权重

                # 误差变小,则更新误差
                if weightedError < minError:

                    minError = weightedError    # 对误差进行更新
                    bestClassEst = predictedVals.copy()     # 存储预测分类结果
                    bestStump['dim'] = i    # 最佳单层,分类特征
                    bestStump['ineq'] = inequal     # 最佳分类
                    bestStump['thresh'] = threshVal     # 最佳分类阈值
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    函数说明:
        基于单层决策树的AdaBoost实现
    :param dataArr: 训练样本数据
    :param classLabels: 样本标签数据
    :param numIt: 迭代次数
    :return:
    """
    # 存储弱分类器
    weakClassArr = []
    m = np.shape(dataArr)[0]    # 矩阵维
    # 初始化权重,样本权重全部初始化为相同的
    D = np.mat(np.ones((m, 1))/m)
    # 每个基分类器的结果进行加权求和
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 迭代numIt次
    for i in range(numIt):
        # 利用D得到的具有最小错误率的单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # 每次迭代后,对基分类器的权重进行更新
        # max取值,为了保证不会发生除0溢出
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        # 最优分类器存储,存储最优的权重
        bestStump['alpha'] = alpha
        # 存储当前迭代,弱分类器的最优情况
        weakClassArr.append(bestStump)

        # 重新计算每个样本的权重D
        expon = np.multiply(-1 * alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        # 对基分类器进行累加求和,权重更新后的情况
        aggClassEst += alpha * classEst
        # 分类错误率
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        # 错误率求和
        errorRate = aggErrors.sum()/m
        print('total error:',errorRate)
        # 错误率变成0,则停止循环
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst

def adaClassify(dataToClass, classifierArr):
    """
    函数说明:
        对AdaBoost算法进行测试
    :param dataToClass:待分类的数据
    :param classifierArr:多个弱分类器(基)
    :return:
    """
    # 待分类数据转变为矩阵
    dataMatrix = np.mat(dataToClass)
    #
    m = np.shape(dataMatrix)[0]
    # 累计分类预测情况
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        print(i)
        aggClassEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                    classifierArr[i]['thresh'],
                                    classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * aggClassEst
        print(aggClassEst)
    return np.sign(aggClassEst)



if __name__ == '__main__':
    dataMat, classLabels = loadSim
    plotData(dataMat, classLabels)

    weakClassArr = adaBoostTrainDS(dataMat, classLabels, 30)
    print('weakClassArr:', weakClassArr)

    result = adaClassify([1, 1], weakClassArr)
    print(result)