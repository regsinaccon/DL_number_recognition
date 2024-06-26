from tensorflow.keras.datasets import mnist
import numpy
import random
from matplotlib import pyplot as plt
import numba
import time



class Model():
    def  __init__(self,Iteration:int,batchsize=512,learning_rate=0.9,momentum=(True,0.5),convolution = True,alpha=0.1):
        assert Iteration>10 ,'Iteration must not less than zero'
        assert batchsize>0 , 'batch size must lager than zero'
        (self.trainImgs,self.trainlable),(self.testImgs, self.testlable) = mnist.load_data()
        self.trainImgs = self.trainImgs.astype('float32') / 255
        self.testImgs = self.testImgs.astype('float32') / 255
        self.testImgs_store = self.testImgs
        self.trainImgs = self.trainImgs.reshape((self.trainImgs.shape[0], -1))
        self.testImgs = self.testImgs.reshape((self.testImgs.shape[0], -1))
        self.Iteration = Iteration
        self.batchsize = batchsize
        self.trainsize = 10000
        self._e = 0.05
        self.w1 = numpy.random.randn(784,128)/numpy.sqrt(784/2)
        self.w2 = numpy.random.randn(129,10)/numpy.sqrt(129/2)
        self.learning_rate = learning_rate
        self.alpha = 1
        self.alpha_change = alpha
        self.history = [0]
        if convolution == True:
            self.trainImgs = self.Convolution(self.trainImgs[:self.trainsize])
            self.testImgs = self.Convolution(self.testImgs)
        self.HasMomentum = momentum[0]
        self.momentum1 = 0
        self.momentum2 = 0
        self.momentumDecay = momentum[1]
        self.update = True
        self.loss = [0]
        self.confidence=[]
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
        TestX = self.testImgs[0:1000]
        TestY = self.testlable[0:1000]
        bias = [85,126,116,107,110,87,87,99,89,94]
        acc = 0
        a = TestX @ self.w1
        b = self.sigmoid(a)
        b1 = numpy.insert(b,0,1,axis=1)
        u = b1 @ self.w2
        p = self.softmax(u)
        numbers = numpy.argmax(p,axis=1)
        for k in range(len(numbers)):
            if numbers[k] == TestY[k]:
                acc += 1000/bias[TestY[k]]
        return acc/100

    def Train(self):
        print('Start training')
        start = time.process_time()
        sigma1_prev = 0
        sigma2_prev = 0
        endindex = self.trainsize-self.batchsize
        for i in range(self.Iteration):
            randinter = random.randint(0,endindex)
            Imgs = self.trainImgs[randinter:randinter+self.batchsize]
            Lables = self.one_hot_encoder(self.trainlable[randinter:randinter+self.batchsize])
            a = Imgs @ self.w1
            b = self.sigmoid(a)
            b1 = numpy.insert(b,0,1,axis=1)
            u = b1 @ self.w2
            yp = self.softmax(u)
            yd = yp - Lables
            bd = b * (1-b) * (yd @ self.w2[1:].T)
            self.cross_entropy(Lables,yp)
            G2 = b1.T @ yd
            G1 = Imgs.T @ bd

            sigma1 = numpy.sqrt(numpy.square(G1*(self.alpha)) + numpy.square((1-self.alpha)*sigma1_prev))
            sigma2 = numpy.sqrt(numpy.square(G2*(self.alpha)) + numpy.square((1-self.alpha)*sigma2_prev))
            sigma1_prev = sigma1
            sigma2_prev = sigma2
            sigma1 += self._e
            sigma2 += self._e
            self.alpha = self.alpha_change
            self.w2 = self.w2 - ((self.learning_rate)/sigma2) * (G2 + self.momentum2)/ self.batchsize
            self.w1 = self.w1 - ((self.learning_rate)/sigma1) * (G1 + self.momentum1)/ self.batchsize
            if self.HasMomentum == True:
                self.momentum1 = (self.momentum1+(G1))*self.momentumDecay
                self.momentum2 = (self.momentum2+(G2))*self.momentumDecay
            Accuracy = self.Predict_acc()
            self.history.append(Accuracy)
            end = time.process_time()
        print(f'Training done after {end - start:.3f} second with accuracy {self.history[-1]:.3f}%')
        return self.history[1:]
    def Store_weight(self):
        numpy.savetxt("weight11.csv",self.w1,delimiter=",")
        numpy.savetxt("weight21.csv",self.w2,delimiter=",")
    def Show(self,option = 'accuracy'):
        if option=='accuracy':
            yaxis = list(range(0,self.Iteration))
            plt.plot(yaxis,self.history[1:])
            plt.yticks(numpy.arange(0, 105, 5))
            plt.xticks(numpy.arange(0,self.Iteration+(self.Iteration/10),int(self.Iteration/10)))
            plt.grid(True)
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy(%)')
            plt.show()
        if option == 'loss':
            yaxis = list(range(0,self.Iteration))
            plt.plot(yaxis,self.loss[1:])
            plt.yticks(numpy.arange(0, numpy.max(self.loss), 0.1))
            plt.xticks(numpy.arange(0,self.Iteration+int(self.Iteration/10),int(self.Iteration/10)))
            plt.grid(True)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()
    def check(self):
        digits = [[] for _ in range(10)]
        for i in range(len(self.testlable)):
            digits[self.testlable[i]].append(i)
        while True:
            index = input('enter an int in range 10:')
            if index == 'end':
                break
            try:
                index = int(index)
                if index<10 and index>=0 and isinstance(index,int):
                    index = digits[index][random.randint(0,len(digits[index])-1)]

                    a = self.testImgs[index] @ self.w1
                    b = self.sigmoid(a)
                    b1 = numpy.insert(b,0,1,axis=0)
                    u = b1 @ self.w2
                    yp = self.softmax(u)
                    number = numpy.argmax(yp)
                    plt.imshow(self.testImgs_store[index], cmap=plt.cm.binary)
                    if number == self.testlable[index]:
                        plt.xlabel(f"True digit: {self.testlable[index]}, Pridict number is: {number}",color='green')            
                    else :
                        plt.xlabel(f"True digit: {self.testlable[index]}, Pridict number is: {number}",color='red')            
                        
                    plt.show()
                else :
                    print('Invalid input')
            except ValueError:
                print("Invalid input")
                print('Enter "end" to terminate the process ')
        print("Process terminated")
    def cross_entropy(self,y_true, y_pred):
        samples = y_true.shape[0]
        logp = - numpy.log(y_pred[numpy.arange(samples), y_true.argmax(axis=1)])
        lossv = numpy.sum(logp)/samples
        self.loss.append(lossv)
    @staticmethod
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
                    if i+steps[j]<784 and i+steps[j]>=0:
                        r[index][i] += Picture[index][i+steps[j]]
        return r

