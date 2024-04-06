from cclass import *


if __name__=='__main__':
    m = Model(300,batchsize=512,learning_rate=0.4,momentum=(True,1),DLR=(True,(0.94,1.05)))
    m.Train()
    m.Store_weight()
    m.Show()
    print(f'Final result accuracy {m.history[-1]}%')