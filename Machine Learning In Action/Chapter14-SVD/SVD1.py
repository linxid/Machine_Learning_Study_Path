import numpy as np

def loadExData():
    return [
        [4,4,0,2,2],
        [4,0,0,3,3],
        [4,0,0,1,1],
        [1,1,1,2,0],
        [2,2,2,0,0],
        [5,5,5,0,0]
    ]

# 假定向量都是列向量
def EculiSim(inA, inB):
    """
    函数说明:
        计算欧几里得距离
    :param inA: 向量A
    :param inB: 向量B
    :return: 欧氏距离
    """
    return 1.0/(1.0 + np.linalg.norm(inA - inB))

def PerasSim(inA, inB):
    """
    函数说明:
        计算person距离
    :param inA: 向量A
    :param inB:
    :return:
    """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(inA, inB, rowvar=0)[0][1]

def CosSim(inA, inB):
    """
    函数说明:
        计算余弦相似度,也就是两个向量的夹角的余弦值
    :param inA:
    :param inB:
    :return: 向量A和B的夹角的余弦值
    """
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    # 利用余弦定理去计算
    return 0.5 + 0.5 * (num/denom)  # 由于范围是[-1,1],将范围变成[0,1]

def standEst(dataMat, user, simMeas, item):
    """
    函数说明:
        对未评分的物体,进行估计评分,虽然user用户对item商品未评分,
        但是根据其他人的评分情况,可以得到两个物品的相似度,
        根据两个物品的相似度来判断
    :param dataMat: 原数据集
    :param user: 用户
    :param simMeas: 相似度计算方法
    :param item: 要估计的物品
    :return: 评分
    """
    n = np.shape(dataMat)[1]
    #
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        # 该用户对第j个物品的评分
        userRating = dataMat[user, j]
        # 如果评分为0,进行下一次循环
        if userRating  == 0:
            continue
        # 统计item物品和j物品都被评分元素的索引
        overlap = np.nonzero(np.logical_and(dataMat[:, item].A > 0,
                                            dataMat[:, j].A > 0))[0]
        # 如果没有均被评分的元素,则相似度是0,两个物物品的相似度
        if len(overlap) == 0:   similarity = 0
        # 否则,计算这两个物品的相似度,
        else:
            similarity = simMeas(dataMat[overlap, item],
                                 dataMat[overlap, j])
        # 输出item和j两个物品的相似度
        print('the %d and %d similarity is : %f' %(item, j, similarity))
        # item物品和每个物品的相似度进行累加
        simTotal += similarity
        # 相似度和评分相乘
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 对评分进行归一化
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=CosSim, estMthod=standEst):
    """
    函数说明:
        预测user用户,对未评分物品的可能评分且取前N个,
    :param dataMat: 原数据
    :param user: 考虑的用户
    :param N: 取前N个
    :param simMeas: 相似度的计算方法
    :param estMthod: 评分估计方法
    :return: 推荐的物品品类和评分
    """
    # 统计未被评分的物品
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everthing.'
    itemsScores = []
    # 遍历每一个伟评分的物品
    for item in unratedItems:
        # 对每一个未被评分的物品进行评分
        estimatedScore = estMthod(dataMat, user, simMeas, item)
        # 保存这些评分值
        itemsScores.append((item, estimatedScore))
        # 对评分进行排序,评分高的在前面
    return sorted(itemsScores, key=lambda jj:jj[1], reverse=True)[:N]

def svdEst(dataMat, user, simMeas, item):
    """
    函数说明:
        对原数据,求SVD然后重新进行预测,
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    """
    n = np.shape(dataMat)[1]
    simTotal = 0.0;ratSimTotal = 0.0
    u, sigma, vT = np.linalg.svd(dataMat)
    # 增加了这部分,求占原数据90%数据量的奇异值
    sig4 = np.mat(np.eye(4) * sigma[:4])
    xformedItems = dataMat.T * u[:, :4]*sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:continue
        similarity = simMeas(xformedItems[item, :].T
                             ,xformedItems[j, :].T)
        print('the %d and %d similarity is :%f'%(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:return 0
    else: return ratSimTotal/simTotal
        

if __name__ == '__main__':
    myMat = np.mat(loadExData())

    print(myMat)
    scores = recommend(myMat, 2)
    print(scores)
