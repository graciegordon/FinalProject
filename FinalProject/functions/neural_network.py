##This python code will take in the number of inputs, size of the hidden layer, and size of output layer. It will train a feed forward neural network

import sys
sys.setrecursionlimit(10000)
import numpy as np
from random import shuffle

class Layer:

    def __init__(self,nodes,inputLayer,nextLayerNode=None):
        self.nodes=nodes #number of nodes in layer
        self.inputLayer=inputLayer #previous network layer
        self.nextLayerNode=nextLayerNode #get number of nodes in next layer
        self.weights=self.computeWeights() #get weight matrix
        self.bias=self.computeBias() #get bias vector
        #self.act=self.activation(self.inputLayer, self.weights) 

    def computeWeights(self):
        #initialize weight matrix if the not the output layer
        if self.nextLayerNode == None:
            weights=None
        else:
            nextnode=self.nextLayerNode
            weights=np.zeros((self.nodes,nextnode)) #make weight matrix
            for i in range(self.nodes): #number input nodes
                for j in range(nextnode): #number of nodes in the next layer
                    #weights[i][j]=np.random.normal(0,0.01**2)
                    weights[i][j]=np.random.normal(0,2)
 
            self.weights = weights
            print('weight',weights)
        return weights

    def computeBias(self):
        #initalize bias vector if not output layer
        if self.nextLayerNode==None:
            bias=None
        else:
            bias=np.zeros([self.nextLayerNode,1]) #number of nodes in next layer
            for i in range(self.nextLayerNode):
                bias[i]=np.random.normal(0,0.01**2)
        print('bias',bias)
        return bias

    def activation(self, inputLayer, x):
        #use the sigmoid function as the activation
        #z=input_weights*input_activations
        z=np.dot(np.transpose(inputLayer.weights),x)+inputLayer.bias
        #print('z',z)
        #sigmoid function
        activation = 1/(1+np.exp(-z))
        return activation

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        return "{0}".format(self.nodes)
        #return "{0} {1}".format(self.nodes, self.act)

def feedforward(network,x):
    #feed forward
    activations=[None]*(len(network)) #store activations from network
    for layer in range(len(network)):

        if layer == 0: #if input layer
            activations[layer]=x

        else: #for every other layers
            activations[layer]=network[layer].activation(network[layer-1],activations[layer-1])
            #wait[layer]=network[layer].weights()
    return activations

def backpropagation(network, x,y):
    #calc delta values from a network for array inputs (x) and outputs (y)
    #return the gradient resluts from the cost fucntion

    #initialize gradient decsent matricies 
    gradw=[None]*(len(network)-1)
    gradb=[None]*(len(network)-1)

    #activations=[None]*(len(network)) #store activations from network
    weights=[None]*(len(network))
    #move to feedforward function

    activations=feedforward(network,x)
    #backprop
    #special calc for the output layer
    a=activations[-1] #output layer activations
    lastact=activations[-1] #store output avctivations
    #sigslope=a*(1-a) #calculate sigmoid'(z) where z is the weights*activations, elementwise multiplication
    #print('a',a)
    #delta=-(y-a)*sigslope #outpus layer delta = activation-output
    #output delta
    delta=(a-y) #last layer delta=activation-output

    #weightsTransposed*delta dot sigslope(z)== activations dot (1-activations)

    #for all other layers 
    for layer in range(len(network)-2,-1,-1): #work backwards though the layers
        a=activations[layer]
        #w=wait[layer]
        #save delta and activations for gradient descent, this is bigDelta
        #bigDelta:=bigdelta+delta(aTranspose))
        #w are the partieal derivative terms, no regularization
        gradw[layer]=np.dot(a,np.transpose(delta))
        gradb[layer]=delta
        
        #if last layer, do not calculate
        if layer == 0:
            break

        sigslope=a*(1-a) #g'(z)=a.*(1-a)
        delta=np.dot(network[layer].weights,delta)*sigslope #delta=weightsTransposed*delta*g'(z)

    return gradw, gradb, lastact

def gradientdescent(network, xmatrix, ymatrix, alpha=0.5, weightDecay=0.9):
    #use logistic regression cost function with regularization
    #weightdeca=lambda
    #J(W,b,x,y)=1/numbertrainingexamples(sum(y*log(h(x))+(1-y)*log(1-(h(x)))))+lambda/2numexampels(sum(sum(sum(weights)^2

    #initialize matricies the same dimensions as the weight/bias of each layer
    #BigDelta to compute partial derivatives, accumulaters 
    deltaW=[None]*(len(network)-1) #store weight of each layer
    deltaB=[None]*(len(network)-1) #store bias vector of each layer

    for layer in range(len(network)-1): #for each layer except input layer
        curnodes=network[layer].nodes
        nextnodes=network[layer].nextLayerNode
        deltaW[layer]=np.zeros([curnodes,nextnodes])
        deltaB[layer]=np.zeros([nextnodes,1])

    #add gradient values of each input
    #D=(1/m)*D+lambda*weights
    inputnodes=np.shape(xmatrix)[0]
    outputnodes=np.shape(ymatrix)[0]
    #print(np.shape(xmatrix))
    #print('outnodes',np.shape(ymatrix))
    #print(len(np.shape(ymatrix)))
    
    #if len(np.shape(ymatrix))==1:
    #    ydim=1
    #else:
    #    ydim=np.shape(ymatrix)[1]
    #print(ydim)
    trainingnum=np.shape(xmatrix)[1]
    #print(inputnodes,outputnodes,trainingnum)    
    #store outputs 
    outputlayer=np.zeros_like(ymatrix)
    outputlayer=outputlayer.astype(float)
    #print(outputlayer)
    #print(xmatrix)
    #print(ymatrix)
    #set columns to inputs
    #loop through training data
    #numOuts=np.shape(outputlayer)[1]
    #ymatrix=np.array(ymatrix)
    #print(ymatrix)
    for i in range(trainingnum):
        x=np.zeros([inputnodes,1])
        #y=np.zeros([outputnodes,1])
        #print('x',x)
        x[:,0]=xmatrix[:,i]
        #y[:,0]=ymatrix[:,i]
        
        y=np.zeros([outputnodes,1])
        y[:,0]=ymatrix[:,i]
    
        #print('x',x)
        #print('y',y)

        #calculate deltas from forward then backpropegation function
        gradW,gradB,lastact=backpropagation(network,x,y)
        ###Why will my activation not add to the final activation matrix???? 
        #print('act',lastact)
        lastact=np.array(lastact)
        fatemp=np.transpose(outputlayer)
        #print('full fat',fatemp)
        acttrans=np.transpose(lastact)
        #print('acttrans',acttrans)
        #print('fatpos',fatemp[i,:])
        
        fatemp[i,:]=acttrans
        #print('edited fa',fatemp[i,:])
        #print('trans last act', np.transpose(lastact))
        
        #fatemp[i,:]=np.transpose(lastact)
        #print('fat2',fatemp) 
        outputlayer=np.transpose(fatemp)
        #print('out',outputlayer)
        #print('act',lastact)
        #print('dW',deltaW)
        #print('gW',gradW)
        #get sum of detlas (gradient) for each layer
        #matrix of D
        #D:=1/m(bigDelta)+lambda*curWeight
        
        #print('datatypes')
        #print('act',lastact.dtype)
        #print('fa',fatemp.dtype)
        #print('outlayer',outputlayer.dtype)

        #calculate bigDelta
        for L in range(len(network)-1):
            #all other deltas
            deltaW[L]=deltaW[L]+gradW[L]
            #bias delta
            deltaB[L]=deltaB[L]+gradB[L]
    
    
    
    #update weight and bias by changing the weights using the learning rate and gradient
    for Layer in range(len(network)-1):
        #update weights with regularization
        newW=network[Layer].weights - alpha*((1/trainingnum)*deltaW[Layer]+weightDecay*deltaW[Layer])
       
        #newW=(1/trainingnum)*(deltaW[Layer]+weightDecay*network[Layer].weights)
        #update bias without regularization
        newB=network[Layer].bias-alpha*((1/trainingnum)*deltaB[Layer])
        
        #newB=(1/trainingnum)*(deltaW[Layer])
        #update arrays
        network[Layer].weights=newW
        network[Layer].bias=newB

    #print(outputlayer)
    #print(deltaW)
    #calculate cost from this round 
    cost=-(1/trainingnum)*sum(sum((ymatrix*np.log10(outputlayer)+(1-ymatrix)*np.log10(1-(outputlayer)))))
    reg=0
    for i in deltaW:
        temp=sum(np.power(i,2))
        #print('temp',temp)
        temp2=sum(temp)
        reg=reg+temp2
    reg=(weightDecay/(2*trainingnum))*reg
    #reg=sum(np.power(deltaW,2))
    totalcost=cost+reg
    #totalcost=-(1/trainingnum)*sum(sum(y*np.log10(outputlayer)+(1-y)*np.log10(1-(outputlayer))+(weightDecay/(2*trainingnum)))*sum(np.power(deltaW,2))
    
    return totalcost, outputlayer, network

def trainNet(samples, labels, inputexample,iters):
    #function that will train a neural network
    #create network
    ins=inputexample
    print('ins',ins)
    networkShape = np.array([ins,int(ins/2),1])
    print('netshape', networkShape)
    inputs = Layer(networkShape[0], None, networkShape[1])
    hidden = Layer(networkShape[1], inputs, networkShape[2])
    outputs = Layer(networkShape[2], hidden)
    newnet = np.array([inputs, hidden, outputs])
    print('in',inputs)
    print('hid',hidden)
    print('out',outputs)
    print('new',newnet)

    #train x times and take lowest cost
    #can make this more sophisticated by training until cost no longer decreases more than a specific amount
    mincost=float('inf')
    for i in range(iters):
        testcost, finalactivation, trainednetwork = gradientdescent(newnet, samples, labels)
        #print('cost',testcost)
        #print('activation',finalactivation)
        if mincost>float(testcost):
            #print('inloop')
            final=finalactivation
            mincost=testcost
    
    return trainednetwork, final, mincost

def testNet(network, samples):
    #given a trained network and new data, it will run the new data through the network
    activations= feedforward(network, samples)
    return activations[-1]

def shufflePosNegs(poslist,neglist):
    #input positive and negative list, shuffle and return
    #shuffle lists
    shuffle(poslist)
    shuffle(neglist)
    #print(poslist[:10])
    #print(neglist[:10])

    #select x examples from each list
    shortposlist=list(np.random.choice(poslist,50,replace=False))
    shortneglist=list(np.random.choice(neglist,50,replace=False))
    print('rand',shortposlist[:10])
    print('rand',shortneglist[:10])
    set1=[]

    #label negative and positive examples
    #0 is negative 1 is positive
    for i in shortposlist:
        #print(i)
        set1.append((i,1))
        #print(set1)
    for i in shortneglist:
        #print(i)
        set1.append((i,0))

    #shuffle set
    shuffle(set1)
    print('set')
    print(set1)
    
    #samples, x to feed into network, labels y to match to the output of the network
    samples=[ x[0] for x in set1 ]
    labels=[ x[1] for x in set1 ]
    label=[] 

    labels=np.array(labels)
    labels=labels.reshape(1, len(labels))
    print(samples)
    print('labels',labels)
    sampleTemp1=[]
    it=0
    for i in samples:
        sampleTemp1.extend(i)
        #print(len(i))
        #it+=1
    #print('len sample',it)
    #print(sampleTemp1)
    sampleTemp=np.array(sampleTemp1)
    print('array')
    print(sampleTemp)
    lensamples=len(samples[0])
    shape=(int(len(samples)),int(len(samples[0])))
    print(shape)
    sampleTemp=sampleTemp.reshape(shape)
    print('reshape')
    print(sampleTemp)
    sampleInput=np.transpose(sampleTemp)
    print('transpose')
    print(sampleInput)
    
    return sampleInput, labels, lensamples

def kfoldStratification(samples, labels):
    #this function will split data into training and test sets X times, 10 is the default
    #first just do one split into training and test sets
    
    assert 1==1


def crossValidation(train, test):
    #this function will train a network then test the network
    assert 1==1

