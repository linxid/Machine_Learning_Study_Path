import matplotlib.pyplot as plt
import numpy as np
def loadDataSet(filename):
    """
    函数说明：
        载入数据
    :param filename:
        文件名
    :return:
        xArr:训练数据特征
        yArr:训练数据标签
    """
    #求数据的特征数
    numFeat = len(open(filename).readline().split('\t')) - 1
    #创建两个孔列表
    xArr = [];yArr=[]
    fr = open(filename)
    #按每行一次读取数据
    for line in fr.readlines():
        #创建列表，保存每行数据
        lineArr = []
        #对每行进行划分，去头去尾，然后按空格划分开
        curLine = line.strip().split('\t')
        #依次读取每个属性的数据，难道不能直接读取吗
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        #按每行一次存储特征数据(前两列)
        xArr.append(lineArr)
        #数据标签值，读取最后一列
        yArr.append(float(curLine[-1]))
    return xArr, yArr

def plotDataSet():
    """
    函数说明：
        绘制数据集的图像，按横纵坐标
    :return:
    """
    xArr,yArr = loadDataSet('ex0.txt')
    n = len(xArr)
    xcord = [];ycord = []
    #按照样本个数循环，依次读取每一行的数据
    #形成x,y坐标
    for i in range(n):
        xcord.append(xArr[i][1]);ycord.append(yArr[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def standRgression(xArr, yArr):
    """
    函数说明：
        求标准回归系数
    :param xArr: 训练数据
    :param yArr: 训练标签
    :return:
        ws：回归系数
    """
    #将训练数据，从数组变成矩阵
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    #按照公式来求解，回归系数
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0:
        print('奇异矩阵，不可逆')
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

def plotRegression():
    #载入数据
    xArr,yArr = loadDataSet('ex0.txt')
    #求回归系数
    ws = standRgression(xArr, yArr)
    #数组转矩阵，便于运算
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    #复制便于处理
    xcopy = xMat.copy()
    #数据与回归系数相乘，得到拟合数据
    yHat = xcopy*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #x数据的第二行为x坐标，第一行全为1
    ax.plot(xcopy[:, 1], yHat, c='red')
    #绘制原数据集的散点图
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
if __name__ == '__main__':
    plotRegression()