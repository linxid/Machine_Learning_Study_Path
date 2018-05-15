import numpy as np

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end=' ')
            else:
                print(0, end=' ')
        print('')

def imgCompress(numSV = 3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print('**********original matrix')
    printMat(myMat, thresh)
    U, sigma, VT = np.linalg.svd(myMat)
    sigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        sigRecon[k, k] = sigma[k]
    reconMat = U[:, :numSV] * sigRecon * VT[:numSV,:]
    print('*************reconstructed matrix using %d singular values*****8'%numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    imgCompress()