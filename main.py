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


batchsize = 10
iteration = 10

w1 = numpy.random.randn(128,784+1)/numpy.sqrt(758/2)
w2 = numpy.random.randn(10,128+1)/numpy.sqrt(129/2)




# print(trainX[0])
for times in range(iteration):
    TrainImg = trainX[times:(times+1)*batchsize]
    TrainNum = trainy[times:(times+1)*batchsize]
    
    for i in range(batchsize):
        # encode the current number
        current_code = one_hot_encoder(TrainNum[i])
        # pass the input vector to the first fatrix
        Append_one(TrainImg[i])
        a = val_pass_weight(TrainImg[i],w1)
        # pass through activation function with bias
        b = sigmoid(a)
        Append_one(b)
        # pass the second fatrix
        u = val_pass_weight(b,w2)
        # pass through activation funtion and u is the estimation
        y = softmax(u) 
        Deviation_Code = Deviation_of(current_code,y)
        print(Deviation_Code)       #
                
