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


def val_pass_weight(input_vector,weight_squar):
    rows = weight_squar.shape[0]
    return_vector = numpy.zeros((rows,))
    for i in range(rows):   
        return_vector[i]=dot(input_vector,weight_squar[i])
    return return_vector

def softmax(vector):
    denominator = 0
    for i in range(len(vector)):
        denominator += math.exp(vector[i])
    for i in range(len(vector)):
        vector[i] = vector[i]/denominator
    return vector 
def sigmoid(vector):
    max_val = max(vector)
    r = numpy.zeros((len(vector),))
    for index in range(len(vector)):
        r[index] = vector[index] - max_val
    for i in range(len(vector)):
        r[i] = 1/(1+math.exp(r[i]*-1))
    return r
def one_hot_encoder(Num):

    encoded = numpy.zeros((10,))
    encoded[Num] = 1
    return encoded
 
def Deviation_of(expected_code,pridict_code):
    return pridict_code-expected_code
 
def Return_Num(vector):
    max_val = -1000
    the_index = 0
    for i in range(len(vector)):
        if vector[i]>max_val:
            max_val = vector[i]
            the_index = i    
    return the_index




# the testx contains only 128 column while w1 has 129 column and so does the things in the hidden layer
# cause dot error
def Predict_acc(testx,testy,w1,w2,test_size):
    acc = 0
    for i in range(test_size):
        Img = numpy.append(testx[i],1)
        a = val_pass_weight(Img,w1)
        a = sigmoid(a)
        b = numpy.append(a,1)
        b = val_pass_weight(b,w2)
        b = softmax(b)
        Number = Return_Num(b)
        if Number == testy[i]:
            acc +=1
    return acc/test_size


@numba.jit(nopython=True)
def Append_one(vector):
    vector = numpy.append(vector,1)
    return vector


def Deviation_b(Deviation_Code,w2,b):
    r = numpy.zeros((129,))
    for i in range(129):
        tmp_sum = 0
        for j in range(10):
            tmp_sum += Deviation_Code[j]*w2[j][i]
        r[i] = b[i]*(1-b[i])*tmp_sum
    return r

def revise_w1():
    pass
