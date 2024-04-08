
from tensorflow.keras.datasets import mnist
import numpy
import random
from matplotlib import pyplot as plt
import numba




class Model():
    def  __init__(self,Iteration,batchsize=512,learning_rate=0.01,momentum=(True,0.5),DLR=(True,(0.94,1.05)),convolution = False):
        (self.trainImgs,self.trainlable),(self.testImgs, self.testlable) = mnist.load_data()
        self.trainImgs = self.trainImgs.astype('float32') / 255
        self.testImgs = self.testImgs.astype('float32') / 255
        self.testImgs_store = self.testImgs
        self.trainImgs = self.trainImgs.reshape((self.trainImgs.shape[0], -1))
        self.testImgs = self.testImgs.reshape((self.testImgs.shape[0], -1))
        self.Iteration = Iteration
        self.batchsize = batchsize
        self.w1 = numpy.random.randn(784,128)/numpy.sqrt(784/2)
        self.w2 = numpy.random.randn(129,10)/numpy.sqrt(129/2)
        self.learning_rate = learning_rate
        self.history = [0]
        if convolution == True:
            self.trainImgs = Convolution(self.trainImgs)
            self.testImgs = Convolution(self.testImgs)
        self.HasMomentum = momentum[0]
        self.momentum1 = 0
        self.momentum2 = 0
        self.momentumDecay = momentum[1]
        if DLR[0] == True:
            self.learning_rate = 0.4
            self.Decay = DLR[1][0]
            self.Enlarge = DLR[1][1]
        else:
            self.Decay = 1
            self.Enlarge = 1
    def one_hot_encoder(self,vector):
        rows = len(vector)
        r = numpy.zeros((rows,10))
        for row in range(rows):
            number = vector[row]
            r[row][number] = 1
        return r
    def sigmoid(self,num):
        return 1/(1+numpy.exp(-num))
    def softmax(self,x):
        x = x.T
        x_max = x.max(axis=0)
        x = x - x_max
        w = numpy.exp(x)
        return (w / w.sum(axis=0)).T
    def Predict_acc(self):
        TestX = self.testImgs[0:500]
        TestY = self.testlable[0:500]
        acc = 0
        a = TestX @ self.w1
        b = self.sigmoid(a)
        b1 = numpy.insert(b,0,1,axis=1)
        u = b1 @ self.w2
        p = self.softmax(u)
        numbers = numpy.argmax(p,axis=1)
        for k in range(len(numbers)):
            if numbers[k] == TestY[k]:
                acc += 1
        return acc/5
    
    def Train(self):
        for i in range(self.Iteration):
            randinter = random.randint(1,59000)
            Imgs = self.trainImgs[randinter:randinter+self.batchsize]
            Lables = self.one_hot_encoder(self.trainlable[randinter:randinter+self.batchsize])
            a = Imgs @ self.w1
            b = self.sigmoid(a)
            b1 = numpy.insert(b,0,1,axis=1)
            u = b1 @ self.w2
            yp = self.softmax(u)
            yd = yp - Lables
            bd = b * (1-b) * (yd @ self.w2[1:].T)

            G2 = b1.T @ yd
            G1 = Imgs.T @ bd
            w1_prev = self.w1
            w2_prev = self.w2
            self.w2 = self.w2 - (self.learning_rate) * (G2 + self.momentum2) / self.batchsize
            self.w1 = self.w1 - (self.learning_rate) * (G1 + self.momentum1)/ self.batchsize
            if self.HasMomentum == True:
                self.momentum1 = (G1)*self.momentumDecay
                self.momentum2 = (G2)*self.momentumDecay
            Accuracy = self.Predict_acc()
            self.history.append(Accuracy)
            if self.history[-1]<self.history[-2]:
                self.w1 = w1_prev
                self.w2 = w2_prev
                self.learning_rate *= self.Decay
            else:
                self.learning_rate *= self.Enlarge 
    def Store_weight(self):
        numpy.savetxt("weight11.csv",self.w1,delimiter=",")
        numpy.savetxt("weight21.csv",self.w2,delimiter=",")
    def Show(self):
        yaxis = list(range(0,self.Iteration))
        plt.plot(yaxis,self.history[1:])
        plt.yticks(numpy.arange(0, 100, 5))
        plt.xticks(numpy.arange(0,self.Iteration+20,20))
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy(%)')
        plt.show()
    def check(self):
        while True:
            index = int(input('enter an index:'))
            if index>1000 or index<0:
                print('process terminated')
                break
            a = self.testImgs[index] @ self.w1
            b = self.sigmoid(a)
            b1 = numpy.insert(b,0,1,axis=0)
            u = b1 @ self.w2
            yp = self.softmax(u)
            number = numpy.argmax(yp)
            plt.imshow(self.testImgs_store[index], cmap=plt.cm.binary)
            plt.xlabel(f"True digit: {self.testlable[index]}, Pridict number: {number}")
            plt.show()

@numba.jit
def Convolution(Picture):
    steps = [-29,-28,-27,-1,0,1,27,28,29]
    kernel = [2,-1,-1,
            -1,2,-1,
            -1,-1,2]
    r = numpy.zeros((Picture.shape[0],784))
    for index in range(Picture.shape[0]):
      for i in range(784):
        for j in range(9):
          if Inrange(i+steps[j]):
            r[index][i] += Picture[index][i+steps[j]]
    return r
@numba.jit
def Inrange(index):
    return index<784 and index >=0
