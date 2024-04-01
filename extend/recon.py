import os
import numpy
from PIL import Image

file1 = './weight1.csv'
file2 = './weight2.csv'

print(os.path.exists(file1)) 
print(os.path.exists(file2))

w1 = numpy.genfromtxt('./weight1.csv',delimiter=',')
w2 = numpy.genfromtxt('./weight2.csv',delimiter=',')
image = Image.open()
print(w1.shape)

print(w2.shape)

def Pridict_Num(file):
    pass