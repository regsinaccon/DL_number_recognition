from cclass import *



m = Model(300,batchsize=512,learning_rate=0.4,momentum=(True,1),DLR=(True,(0.94,1.05)))
t = m.Train()
yaxis = list(range(0,300))
plt.plot(yaxis,t[1:])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
print(t[-1])