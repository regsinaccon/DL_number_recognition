import matplotlib.pyplot as plt
import numpy 
from tensorflow.keras.datasets import mnist
from tools import Return_Num
(trainX, trainy), (testX, testy) = mnist.load_data()

w1 = numpy.genfromtxt('./weight1.csv',delimiter=',')
w2 = numpy.genfromtxt('./weight2.csv',delimiter=',')
testXX = testX.reshape((testX.shape[0], -1))

def display_image(index):
    plt.figure()
    plt.imshow(testX[index], cmap=plt.cm.binary)
    plt.xlabel(f"True digit: {testy[index]}, Pridict number: {Return_Num(testXX[index],w1,w2)}")
    plt.show()

 
while True:
    index = int(input("enter a number:"))
    if index<1000:
        display_image(index)
    else :
        break