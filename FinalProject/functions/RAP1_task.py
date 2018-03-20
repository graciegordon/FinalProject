#This code will take train a neural network to detect RAP1 sites

import neural_network as nn
import utils as util
from random import shuffle
import numpy as np

#read in sites
posfile='/Users/student/Documents/Algorithms/FinalProject/data/rap1-lieb-positives.txt'
negfile='/Users/student/Documents/Algorithms/FinalProject/data/yeast-upstream-1k-negative.fa'
testfile='/Users/student/Documents/Algorithms/FinalProject/data/rap1-lieb-test.txt'
poslist=util.read_pos(posfile)
poslist=util.seq_encode(poslist)
#print(poslist)
#print(posmat)
#print('neg')

neglist=util.read_fasta(negfile)
print('negs',neglist[:10])

shortneg=[]
for i in neglist:
    #TODO adapt so this is a random slice of the negative
    shortneg.append(str(i[3:20]))

#encode neg
#neglist=util.seq_encode(shortneg)

neglist=shortneg
neglist=util.seq_encode(neglist)
print('neglist',neglist[:10])
#sample x random seqs from positive and x from the negative to make sample
#keep track of which samples come from each group
#start with small training set
#shuffle
shuffle(poslist)
shuffle(neglist)
print(poslist[:10])
print(neglist[:10])

shortposlist=list(np.random.choice(poslist,50,replace=False))
shortneglist=list(np.random.choice(neglist,50,replace=False))
print('rand',shortposlist[:10])
print('rand',shortneglist[:10])
set1=[]
#0 is negative 1 is positive
for i in shortposlist:
    #print(i)
    set1.append((i,1))
    #print(set1)
for i in shortneglist:
    #print(i)
    set1.append((i,0))

shuffle(set1)
print('set')
print(set1)

#samples, x to feed into network, labels y to match to the output of the network
samples=[ x[0] for x in set1 ]
labels=[ x[1] for x in set1 ]
label=[]
#for l in labels:
#    label.append(list(l))
#print(label)
#label=np.array(label)
#print('label',label)
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

shape=(int(len(samples)),int(len(samples[0])))
print(shape)
sampleTemp=sampleTemp.reshape(shape)
print('reshape')
print(sampleTemp)
sampleInput=np.transpose(sampleTemp)
print('transpose')
print(sampleInput)
#for i in sampleInput:
#    print(i)


##create network
ins=len(samples[0])
networkShape = np.array([ins,int(ins/2),1])
inputs = nn.Layer(networkShape[0], None, networkShape[1])
hidden = nn.Layer(networkShape[1], inputs, networkShape[2])
outputs = nn.Layer(networkShape[2], hidden)
newnet = np.array([inputs, hidden, outputs])
print('in',inputs)
print('hid',hidden)
print('out',outputs)
print('new',newnet)

#TODO implement Stratified Kfold CV

#print(samples[0])
mincost=float('inf')
testcost=5
for i in range(100):
    #while mincost>=float(testcost):
    testcost, finalactivation, trainednetwork = nn.gradientdescent(newnet, sampleInput, labels)
    print('cost',testcost)
    print('activation',finalactivation)
    if mincost>float(testcost):
        print('inloop')
        final=finalactivation
        mincost=testcost
print('final cost',mincost)
print('final',final)
print('rounded',final.round(decimals=2))

'''
networkShape = np.array([68,30,1])
inputs = nn.Layer(networkShape[0], None, networkShape[1])
hidden = nn.Layer(networkShape[1], inputs, networkShape[2]) # hidden layer with 3 nodes, takes in inputs layer
outputs = nn.Layer(networkShape[2], hidden)
newnet = np.array([inputs, hidden, outputs])

print('in',inputs)
print('hid',hidden)
print('out',outputs)
print('new',newnet)

#Stratefied Kforld CV, imbalanced dataset



#print(identityinput3)
print('start')
print(identityinput8)
#x = np.array([[1],[0],[0],[0],[0],[0],[0],[0]])
mincost=float('inf')
testcost=5
for i in range(100000):
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
'''
#testlist=util.read_pos(testfile)
#print('test',testlist)
