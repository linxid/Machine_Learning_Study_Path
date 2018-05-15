import re
import numpy as np

def textParse(bigString):
    """
    函数说明：
        将字符串转换为字符列表，也就是实现文本切分
    :param bigString:
    :return:
    """
    listOfTokens = re.split(r'\W*', bigString)                              # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            # 去除少于两个字符的字符串，并且将其变成小写

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

# ---------------------------------------------------------------------
# 注意两个的区别，一个是词集模型，一个是词袋模型
# 词集模型：只统计一个词是否出现
# 词袋模型：统计每个词出现的次数
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

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
# ------------------------------------------------------------------------
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' %i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 在trainingSet中随机选择10个到testSet
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))        # 产生0-50的随机数，且10次不同
        testSet.append(trainingSet[randIndex])      # 对testSet赋值，将随机数据赋给testSet
        del(trainingSet[randIndex])         # 删除trainingSet中相对应的数据
    # ------------------------------------------------------------------------------------------------
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