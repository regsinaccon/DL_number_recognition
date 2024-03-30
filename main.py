import numpy 
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tools import *

 
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

# normalize pixel values to be between 0 and 1
trainX = trainX.astype('float32') / 255
testX = testX.astype('float32') / 255

# reshape the trainx and testx from 28x28 to 784
trainX = trainX.reshape((trainX.shape[0], -1))
testX = testX.reshape((testX.shape[0], -1))




# check the shapes
print("Training data shape:", trainX.shape)
print("Testing data shape:", testX.shape)


batchsize = 512
iteration = 50
learning_rate = 0.01


w1 = numpy.random.randn(784,128)/numpy.sqrt(784/2)
w2 = numpy.random.randn(129,10)/numpy.sqrt(129/2)




# print(trainX[0])
for times in range(iteration):
    print(f'accuracy is {Predict_acc(testX,testy,w1,w2,100)*100:.2f}percent')
    TrainImg = trainX[times:(times+1)*batchsize]
    TrainNum = trainy[times:(times+1)*batchsize]
    yt = one_hot_encoder(TrainNum)
    a = TrainImg @ w1
    b = sigmoid(a)
    b1 = numpy.insert(b, 0, 1, axis=1)   
    u = b1 @ w2                         
    yp = softmax(u)
    yd = yp - yt                      
    bd = b * (1-b) * (yd @ w2[1:].T)  

    w2 = w2 - learning_rate * (b1.T @ yd) / batchsize   
    w1 = w1 - learning_rate * (TrainImg.T @ bd) / batchsize
     



