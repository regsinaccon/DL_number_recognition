from cclass import *
import numpy

if __name__=='__main__':
    m = Model(300,batchsize=512,learning_rate=0.01,momentum=(True,0.5),DLR=True,convolution=True)
    m.Train()
    m.Show('loss') #loss or accuracy
    m.check()