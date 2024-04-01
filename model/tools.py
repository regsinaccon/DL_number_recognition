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




@numba.jit
def Deviation_b(Deviation_Code,w2,b):
    r = numpy.zeros((129,))
    for i in range(129):
        tmp_sum = 0
        for j in range(10):
            tmp_sum += Deviation_Code[j]*w2[j][i]
        r[i] = b[i]*(1-b[i])*tmp_sum
    return r
@numba.jit
def revise_w2(w2_chage,b,Deviation_Code):
    for i in range(10):
        for j in range(1,129):
            w2_chage[i][j] +=b[j]*Deviation_Code[i]
@numba.jit
def revise_w1(w1_chage,Img,D_b):
    for i in range(128):
        for j in range(785):
            w1_chage[i][j] += Img[j]*D_b[i]


def Rate_editor(Gradinet,Last_Gradient,alpha = 0.3):
    Lg = numpy.square(Last_Gradient)
    g = numpy.square(Gradinet)
    return numpy.sqrt(alpha*numpy.sum(Lg)+(1-alpha)*numpy.sum(g))
@numba.jit
def Inrange(index):
    if index <= 784 and index >= 0:
        return True
    else:
        return False


@numba.jit
def Convolution(Img):
    steps = [-29,-28,-27,-1,0,1,27,28,29]
    kernel = [-1,-1,-1,
          -1,8,-1,
          -1,-1,-1]
    r = numpy.zeros((Img.shape[0],784))
    for index in range(Img.shape[0]):
        for i in range(784):
            for j in range(9):
              if Inrange(i+steps[j]):
                r[index][i] += Img[index][i+steps[j]]*kernel[j]
              else:
                continue
    return r