import numpy as np
from functools import reduce
import re
import random

def loadDataSet():
    """
    函数说明：
        创建实验数据
    参数：
        无
    :return:
        postingList--实验样本切分词条
        classVec--类别标签向量
    """
    postingList=[
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def setOfWord2Vec(vocabList, inputSet):
    """
    函数说明：
        将文本转换成词向量
    参数：
    :param vocabList: 用于参考的字典或者说词汇表，文本词的总和
    :param inputSet: 输入文本
    :return:
    """
    # 建立一个词向量，用于存储词文本，相当于某种索引
    # 将一个文本中的几个词，对应到词典中，相应的位置设为1
    returnVec = [0] * len(vocabList)
    # 遍历输入文本
    for word in inputSet:
        # 如果这个单词在字典里面，就令相应的位置为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            # 否则指出，该单词不在字典中
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec

def createVocabList(dataSet):
    """
    函数说明：
        创建字典，通过将多个文本组合，然后去重，
    参数：
    :param dataSet: 数据集
    :return: 字典
    """
    # 因为去重，所以使用set数据类型
    vocabSet = set([])
    for document in dataSet:
        # 因为是集合，所以使用此操作符，即可实现并集操作
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def trainNB0(trainMatrix, trainCategory):
    """
    函数说明：
        该函数用于计算各个类别，不同属性的每个属性值的概率，
        为了计算后验概率，需要先验概率和条件概率
    参数：
    :param trainMatrix:所有文本转换成的词向量
    :param trainCategory:样本类别
    :return:
    """
    numTrainDocs = len(trainMatrix)     # 文本词向量的文档个数
    numWords = len(trainMatrix[0])      # 文本词向量的长度
    # 因为共两类，侮辱为1，求和后除以总样本数，即为先验概率，每个类别所占的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)       #计算先验概率，
    # 初始化分子，各个词出现的次数
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 初始化分母，每个类别的总词数
    # 不取对数，且不修正
    # p0Denom = 0.0
    # p1Denom = 0.0
    # 对数据取对数同时，做拉普拉斯修正
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历所有文档，
    for i in range(numTrainDocs):
        # 计算属于侮辱性文档的概率
        if trainCategory[i] == 1:
            # 一旦是，侮辱性文档，相应的字数加一，
            p1Num += trainMatrix[i]
            # 该类比的总词数
            p1Denom += sum(trainMatrix[i])
        else:
            # 非侮辱性文档，字数和总词数的统计
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 没有取对数，会出现0概率，会使得最后算的后验概率为0，
    # p1Vect = p1Num/p1Denom
    # p0Vect = p0Num/p0Denom
    # 取对数，避免下溢，同时，平滑做拉普拉斯修正，
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 未取对数
    # p1 = reduce(lambda x,y:x*y, vec2Classify*p1Vec) * pClass1
    # p0 = reduce(lambda x,y:x*y, vec2Classify*p0Vec) * (1.0 - pClass1)
    # 取对数，所以相乘变成相加
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    """
    函数说明：
        测试数据，拿两个文档（文本）测试是否是侮辱性语言
    :return:
    """
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my','dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')

    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')

def textParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' %i,'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('分类错误的测试集：',docList[docIndex])
    print('错误率：%.2f%%'%(float(errorCount)/len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
