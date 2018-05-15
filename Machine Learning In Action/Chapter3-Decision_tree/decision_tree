from math import log
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import operator
#------------------------------------------------------------------


def createDataSet():
    """
    函数说明：创建测试数据集
    参数：无
    return: dataSet-数据集
        labels-分类属性
    """
    dataSet = [
        [0,0,0,0,'no'],
        [0,0,0,1,'no'],
        [0,1,0,1,'yes'],
        [0,1,1,0,'yes'],
        [0,0,0,0,'no'],
        [1,0,0,0,'no'],
        [1,0,0,1,'no'],
        [1,1,1,1,'yes'],
        [1,0,1,1,'yes'],
        [1,0,1,2,'yes'],
        [2,0,1,2,'yes'],
        [2,0,1,1,'yes'],
        [2,1,0,1,'yes'],
        [2,1,0,2,'yes'],
        [2,0,0,0,'no']
    ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet,labels
#-------------------------------------------------------------------------


def calShannoEnt(dataSet):
    """
    函数说明：
        计算数据集的香农熵
    param ：dataSet-数据集
    return:  shannonEnt-香农熵
    """
    numEntires = len(dataSet)   #数据集行数
    labelCounts = {}    #该字典用于保存每个标签出现的次数
    for featVec in dataSet:
        currentLabel = featVec[-1]  #for循环，来处理每一个数据的标签
        if currentLabel not in labelCounts.keys():  #对字典进行初始化
            labelCounts[currentLabel] = 0   #令两类初始值分别为0
        labelCounts[currentLabel] += 1  #统计数据

    shannonEnt = 0.0
    for key in labelCounts:     #根据公式，利用循环求香农熵
        prob = float(labelCounts[key])/numEntires
        shannonEnt -= prob*log(prob,2)
    return shannonEnt
#-------------------------------------------------------------
def splitDataSet(dataSet,axis, value):
    """
    函数说明：
        对数据集根据某个特征进行划分
    :param dataSet: 被划分的数据集
    :param axis: 划分根据的特征
    :param value: 需要返回的特征的值
    :return: 无
    """
    retDataSet = []     #建立空列表，存储返回的数据集
    for featVec in dataSet:     #遍历数据，一个对象一个对象的进行操作
        if featVec[axis] == value:      #找到axis特征，等于value值得数据
            reducedFeatVec = featVec[:axis]    #选出去除axis特征，只保留其他特征的数据
            reducedFeatVec.extend(featVec[axis+1:])     #保留后面的
            retDataSet.append(reducedFeatVec)       #对数据进行连接，按照列表形式
    return retDataSet   #返回划分后被取出的数据集，即满足条件的数据
#---------------------------------------------------------------------------
def chooseBestFeaturesToSplit(dataSet):
    """
    函数说明：
        选择最佳的分类属性
    :param ：dataSet-数据集
    :return: bestFeature-最佳划分属性
    """
    numFeatures = len(dataSet[0]) - 1       #特征数量，去除最后一个标签值
    baseEntropy = calShannoEnt(dataSet)     #计算原始数据的香农熵
    bestInfoGain = 0.0      #信息增益初始化
    bestFeature = -1        #最有特征索引
    for i in range(numFeatures):        #遍历所有特征
        #该特征的所有值，组成的列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)      #创建集合，元素不可重复
        newEntropy = 0.0        #经验条件熵
        for value in uniqueVals:        #计算信息增益
            #根据第i特征，value值对数据进行划分
            subDataSet = splitDataSet(dataSet, i, value)
            #求划分后数据的经验条件熵
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calShannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #求划分前后的信息增益
        print("第%d个特征的增益是%.3f" %(i,infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
#------------------------------------------------------------------------
def majorityCnt(classList):
    """
    函数说明：
        当划分结束，遍历所有特征，但是依然不能彻底划分数据时
        将这多个类别中，出现次数最多的作为该类别
    :param classList-类别列表
    :return: sortedClassCount[0][0]-类别
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1       #统计出现各个类别出现的次数
    #对类别列表进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]
#-------------------------------------------------------------------------
def createTree(dataSet, labels, featLabels):
    """
    函数说明：
        创建决策树
    :param dataSet-训练集
    :param labels-训练集数据标签
    :param featLabels:
    :return:
    """
    classList = [example[-1] for example in dataSet]        #取分类标签(是否放贷:yes or no)
    #注意条件，对列表的某一类计数，若为列表长度，则类别完全相同
    if classList.count(classList[0]) == len(classList):      #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:         #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeaturesToSplit(dataSet)       #最优特征
    bestFeatLabel = labels[bestFeat]        #最优特征的标签
    featLabels.append(bestFeatLabel)        #
    myTree = {bestFeatLabel:{}}     #根据最优特征的标签建立决策树
    del(labels[bestFeat])       #删除已经使用的特征标签
    #训练集中，最优特征所对应的所有数据的值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)        #去掉重复特征
    for value in uniqueVals:        #遍历特征创建决策树
        #通过此句实现字典的叠加
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),labels,featLabels)
    return myTree
#---------------------------------------------------------------------------
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

dataSet, labels = createDataSet()
featLabels = []
myTree = createTree(dataSet, labels, featLabels)
testVec = [0, 1]
result = classify(myTree, featLabels, testVec)
if result == 'yes':
    print('放贷')
if result == 'no':
    print('不放贷')
