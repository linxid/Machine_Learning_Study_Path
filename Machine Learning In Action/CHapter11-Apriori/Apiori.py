import numpy as np

def loadDataSet():
    """
    函数说明:
        载入简单的数据
    :return:
    """
    return [[1, 3, 4],[2, 3, 5], [1, 2, 3, wsenmnmm5,],[2, 5]]

def createC1(dataSet):
    """
    函数说明:
        创建第一个满足条件的集合,集合大小为1的所有候选项的集合
    :param dataSet: 样本数据
    :return:
    """
    C1 = []     # 存储大小为1的项的集合
    for transaction in dataSet:     #依次循环
        for item in transaction:      # 寻找不重复的大小为1的项
            if item not in C1:
                C1.append([item])

    C1.sort()      # 对集合进行排序,数字就是
    return map(frozenset, C1)   # 将list转成frozenset类型,便于处理

def scanD(D, Ck, minSupport):
    """
    函数说明:
        根据C1生成L1,也就是满足最小支持度要求的集合的超集,项的大小是2
    :param D: 数据集
    :param Ck: 满足条件的集合大小为k的项的结合
    :param minSupport: 最小支持度
    :return:
    """
    ssCnt = {}  # 统计包含项k的超集的个数
    for tid in D:   # 遍历数据
        for can in Ck:      # 遍历Ck的所有候选值
            # 字典键值为Ck,统计出现的次数,
           if can not in ssCnt:
               ssCnt[can] = 1
            # 未出现则赋1,出现则累加
           else:
               ssCnt[can] += 1
    # 总的数据集个数
    D = list(D)
    numItems = float(len(D))
    #
    retList = []
    supportData = {}    # 统计每个类别的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 存储满足支持度要求的集合
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet, minSupport = .5):
    C1 = createC1(dataSet)

    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supk = scanD(dataSet, Ck, minSupport)
        supportData.update(supk)
        L.append(Lk)
        k += 1
    return L, supportData



if __name__ == "__main__":
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet)
    print(list(L))
