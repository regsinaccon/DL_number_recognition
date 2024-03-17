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

# def split_the_squar(squar):
#     rows = squar.shape[0]
#     splitted_squar = numpy.zeros(rows)

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
# 
def Deviation_of(expected_code,pridict_code):
    return pridict_code-expected_code
# 

# def Pridict_acc()



# maybe not so useful in b deviation
def Sigma(*args):
    sum = 0
    for num in args:
        sum += num
    return sum



@numba.jit(nopython=True)
def Append_one(vector):
    vector = numpy.append(vector,1)
    return vector

'''
unfinished function
'''
def Deviation_b(Deviation_Code,w2):
    r = numpy.zeros((129,))
    for i in range(129):
        for j in range(10):
            pass
