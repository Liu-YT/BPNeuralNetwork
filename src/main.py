# coding:utf-8 
from io_helper import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime


# 图片训练集
train_images_idx3_ubyte_file = './data/train-images.idx3-ubyte'

# 标签训练集
train_labels_idx1_ubyte_file = './data/train-labels.idx1-ubyte'

# 图片测试集
test_images_idx3_ubyte_file = './data/t10k-images.idx3-ubyte'

# 标签测试集
test_labels_idx1_ubyte_file = './data/t10k-labels.idx1-ubyte'

# 展示图片
def draw_img(img):
    _, ax = plt.subplots(
        sharex = True,
        sharey = True,
    )
    ax.imshow(img.reshape(28,28), cmap="Greys", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

class BP:
    def __init__(self):
        self.weightIH, self.weightHO = [], []
        self.minist = load_mnist(train_images_idx3_ubyte_file, train_labels_idx1_ubyte_file, test_images_idx3_ubyte_file, test_labels_idx1_ubyte_file)
    
    '''
    生成初始的权值矩阵
    经验公式: 权值的初始值应选为均匀分布的经验值，在(-2.4/F, 2.4/F)之间
    其中F为所连单元的输入端个数
    '''
    def weightMatrixGen(self, weight, row, col):
        temp = 2.4 / row
        for i in range(row):
            for _ in range(col):
                weight[i].append(random.uniform(-temp, temp)) 
    
    '''
    Sigmoid函数
    '''
    def sigmoid(self, nodes):
        return 1.0 / (1.0 + np.exp(-1 * nodes))
    
    '''
    为避免不收敛，提高学习速率，期望输出数相应小一些
    '''
    def expected(self, label):
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = 0.01
            else:
                label[i] = 0.99
        return label
    
    def train(self):
        # 输入层、隐含层、输出层的结点数
        self.inputLayer, self.hidLayer, self.outputLayer = 784, 30, 10

        # 学习速率
        rate = 0.1

        # 输入层与隐含层之间的权值矩阵
        self.weightIH = [[] for i in range(self.inputLayer)]
        self.weightMatrixGen(self.weightIH, self.inputLayer, self.hidLayer)
        self.weightIH = np.array(self.weightIH)

        # 隐含层与输入层之间的权值矩阵
        self.weightHO = [[] for i in range(self.hidLayer)]
        self.weightMatrixGen(self.weightHO, self.hidLayer, self.outputLayer)
        self.weightHO = np.array(self.weightHO)

        '''
        BP训练网络方式
        顺序方式，为每输入一个训练样本修正一次权值的方式
        '''
        print('Start train')
        startTime = datetime.datetime.now()
        n = 0
        while n < len(self.minist[0][0]):
            '''
            激励输出
            '''
            # 输入层输出
            iLayerOut = self.sigmoid(np.array([self.minist[0][0][n]]))
            # 隐含层输出
            hLayerOut = self.sigmoid(iLayerOut @ self.weightIH)
            # 输出层输出
            oLayerOut = self.sigmoid(hLayerOut @ self.weightHO)

            '''
            期望
            '''
            EO = oLayerOut * (1 - oLayerOut) * (self.minist[0][1][n] - oLayerOut)
            EH = hLayerOut * (1 - hLayerOut) * (self.weightHO @ EO.T).T

            '''
            权值修正
            '''
            self.weightHO = self.weightHO + rate * hLayerOut.T @ EO
            self.weightIH = self.weightIH + rate * iLayerOut.T @ EH
            n += 1

        #long running
        endTime = datetime.datetime.now()
        print('End train')
        print(str((endTime - startTime).seconds) + 's')

    def test(self):
        count, n = 0, 0
        while n < len(self.minist[1][0]):
            iLayerOut = self.sigmoid(np.array([self.minist[1][0][n]]))
            hLayerOut = self.sigmoid(iLayerOut @ self.weightIH)
            oLayerOut = self.sigmoid(hLayerOut @ self.weightHO)
            if np.argmax(self.minist[1][1][n]) == np.argmax(oLayerOut):
                count = count + 1
            n = n + 1
        print("正确率为: " + str(count * 100.0 / len(self.minist[1][0])) + "%")



if __name__ == "__main__":
    bp = BP()
    bp.train()
    bp.test()