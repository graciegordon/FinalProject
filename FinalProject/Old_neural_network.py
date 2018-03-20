##This python code will take in the number of inputs, size of the hidden layer, and size of output layer. It will train a feed forward neural network

import numpy as np

def Sigmoid(x):
    #calculate sigmoid based on a given x array
    x=np.array(x)
    negX=x*(-1)
    sig=1/(1+np.exp(negX))
    return sig

def Initialize_Weights(x,y,z):
    #take in weight dimensions and return randomly initialized matrix
    weightMat=np.random.uniform(low=-1.0, high=1.0, size=(x,y,z))
    print(weightMat)

    return weightMat
    
def Build_Network(inputs, hiddenUnits, hiddenLayers, outputs):
    #take in input layer, and size of desired hidden and output layer
    #create a weight matrix
    #add bias unit to input layer
    inputs=[1]+inputs
    sizeInput=len(inputs)
    
    weights=Initialize_Weights(1,sizeInput, hiddenLayers+1)

    a=inputs
    for i in range(hiddenLayers+1):
        a=Sigmoid(np.multiply(a,weights[0][:hiddenUnits]))

    print(a)

Build_Network(['001','0001','1000','0001','1000','0001','1000','0100'],3,1,8)


