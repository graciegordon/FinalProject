from functions import neural_network as nn
from functions import utils
import numpy as np

##testing autoencoder
networkShape = np.array([8,3,8])
inputs = nn.Layer(networkShape[0], None, networkShape[1])
hidden = nn.Layer(networkShape[1], inputs, networkShape[2]) # hidden layer with 3 nodes, takes in inputs layer
outputs = nn.Layer(networkShape[2], hidden)
newnet = np.array([inputs, hidden, outputs])

print('in',inputs)
print('hid',hidden)
print('out',outputs)
print('new',newnet)

#identityinput3 = np.identity(3)
identityinput8 = np.identity(8)

#print(identityinput3)
print('start')
print(identityinput8)
#x = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
mincost=float('inf')
testcost=5
for i in range(10000):
    #while mincost>=float(testcost):
    testcost, finalactivation, trainednetwork = nn.gradientdescent(newnet, identityinput8, identityinput8)
    #print('cost',testcost)
    #print('activation',finalactivation)
    if mincost>float(testcost):
        #print('inloop')
        final=finalactivation
        mincost=testcost
print('final cost',mincost)
print('final',final)
print('rounded',final.round(decimals=2))
