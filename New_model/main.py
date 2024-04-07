from cclass import *


if __name__=='__main__':
    m = Model(300,batchsize=512,learning_rate=0.01,momentum=(True,0.5),DLR=(True,(0.94,1.05)),convolution=True)
    m.Train()
    m.Store_weight()
    print(f'Final result accuracy {m.history[-1]}%')
    m.Show()
    