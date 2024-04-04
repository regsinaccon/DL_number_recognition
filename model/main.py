
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

trainX = Convolution(trainX)
testX = Convolution(testX)



# check the shapes
print("Training data shape:", trainX.shape)
print("Testing data shape:", testX.shape)


batchsize = 512
iteration = 117
learning_rate = 0.27
Decay = 0.95
w1 = numpy.random.randn(784,128)/numpy.sqrt(784/2)
w2 = numpy.random.randn(129,10)/numpy.sqrt(129/2)
w1_prev = w1
w2_prev = w2


momentum1 = 0
momentum2 = 0
loss = [0]
history = [0]
yaxis = list(range(0,iteration))
# print(trainX[0])
for times in range(iteration):

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

    G2 = b1.T @ yd
    G1 = TrainImg.T @ bd
    w1_prev = w1
    w2_prev = w2
    w2 = w2 - (learning_rate) * (G2 + momentum2) / batchsize
    w1 = w1 - (learning_rate) * (G1 + momentum1 )/ batchsize
    momentum1 = (G1)
    momentum2 = (G2)
    acc = Predict_acc(testX,testy,w1,w2,500)
    history.append(acc)
    if history[-1]<history[-2]:
        w1 = w1_prev
        w2 = w2_prev
        learning_rate *= Decay



numpy.savetxt("weight1.csv",w1,delimiter=",")
numpy.savetxt("weight2.csv",w2,delimiter=",")

plt.plot(yaxis,history[1:])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
print(acc)
print(learning_rate)
