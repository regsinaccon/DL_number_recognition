import os
import numpy
from PIL import Image
from tools import Convolution ,sigmoid , softmax ,Return_Num
import subprocess

def Pridict_Num():

    draw = './drawing.png'
    w1 = numpy.genfromtxt('./weight1.csv',delimiter=',')
    w2 = numpy.genfromtxt('./weight2.csv',delimiter=',')
    image = Image.open(draw)
    new_size = (28,28)
    image.thumbnail(new_size)
    Img = numpy.array(image)
    Img = abs(Img-255)
    Img = Img.astype('float32')/255
    Img = Img.reshape(784)
    Num = Return_Num(Img,w1,w2)
    print(Num)
    return Num
