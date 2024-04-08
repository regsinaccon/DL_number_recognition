from cclass import *
import numpy

if __name__=='__main__':
    m = Model(300,batchsize=512,learning_rate=0.01,momentum=(True,0.5),DLR=(True,(0.94,1.05)),convolution=True)
    m.Train()
    m.Show()
    while True:
        index = int(input('enter a index:'))
        m.check(index)