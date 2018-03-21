#This code will take train a neural network to detect RAP1 sites

import neural_network as nn
import utils as util
from random import shuffle
import numpy as np

#read in sites
posfile='/Users/student/Documents/Algorithms/Alg_final_project/FinalProject/data/rap1-lieb-positives.txt'
negfile='/Users/student/Documents/Algorithms/Alg_final_project/FinalProject/data/yeast-upstream-1k-negative.fa'
testfile='/Users/student/Documents/Algorithms/Alg_final_project/FinalProject/data/rap1-lieb-test.txt'
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

sampleInputs,labels,samplelength=nn.shufflePosNegs(poslist,neglist)

trainednet,final,mincost=nn.trainNet(sampleInputs, labels, samplelength,1000)


print('final cost',mincost)
print('final',final)
print('labels', labels)
print('rounded',final.round(decimals=2))

#testlist=util.read_pos(testfile)
#print('test',testlist)
