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




def Return_Num(Img,w1,w2):
    a = Img @ w1
    b = sigmoid(a)
    b1 = numpy.insert(b, 0, 1)
    u = b1 @ w2
    yp = softmax(u)
    Number = numpy.argmax(yp)
    return Number



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


def Rate_editor(Gradinet,Last_Gradient,alpha = 0.3):
    Lg = numpy.square(Last_Gradient)
    g = numpy.square(Gradinet)
    return numpy.sqrt(alpha*numpy.sum(Lg)+(1-alpha)*numpy.sum(g))
@numba.jit
def Inrange(index):
    if index < 784 and index >= 0:
        return True
    else:
        return False


@numba.jit
def Convolution(Img):
    steps = [-29,-28,-27,-1,0,1,27,28,29]
    kernel = [2,-1,-1,
            -1,2,-1,
             -1,-1,2]
    r = numpy.zeros((Img.shape[0],784))
    for index in range(Img.shape[0]):
        for i in range(784):
            for j in range(9):
              if Inrange(i+steps[j]):
                r[index][i] += Img[index][i+steps[j]]*kernel[j]
              else:
                continue
    return r


def one_con(Img):
    steps = [-29,-28,-27,-1,0,1,27,28,29]
    kernel = [2,-1,-1,
            -1,2,-1,
             -1,-1,2]
    r = numpy.zeros((784,))
    for i in range(784):
        for j in range(9):
            if Inrange(i+steps[j]):
                r[i] +=Img[i+steps[j]]*kernel[j]
                # r[index][i] += Img[index][i+steps[j]]*kernel[j]
            else:
                continue
    return r