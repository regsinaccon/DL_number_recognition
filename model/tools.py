import numpy
import math 
import random
import numba
@numba.jit
def dot(vector1,vector2):
    sum = 0
    for column in range(len(vector1)):
        sum += vector1[column]*vector2[column]
    return sum




# pp soft mss
def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = numpy.exp(x)
    return (w / w.sum(axis=0)).T
    
def sigmoid(num):
    return 1/(1+numpy.exp(-num))
def one_hot_encoder(muti_vector):
    rows = len(muti_vector)
    r = numpy.zeros((rows,10))
    for row in range(rows):
        number = muti_vector[row]
        r[row][number] = 1
    return r
    
 
def Deviation_of(expected_code,pridict_code):
    return pridict_code-expected_code
 

@numba.jit
def Return_Num(vector):
    max_val = -1000
    the_index = 0
    for i in range(len(vector)):
        if vector[i]>max_val:
            max_val = vector[i]
            the_index = i    
    return the_index




def Predict_acc(testx,testy,w1,w2,test_size):
    acc = 0
    for i in range(test_size):
        Img = testx[i]
        a = Img @ w1
        b = sigmoid(a)
        b1 = numpy.insert(b, 0, 1)
        u = b1 @ w2
        yp = softmax(u)
        Number = numpy.argmax(yp)

        if Number == testy[i]:
            acc +=1
    return acc/test_size







