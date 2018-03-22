#This code will take train a neural network to detect RAP1 sites

import neural_network as nn
import utils as util
from random import shuffle
import numpy as np
import random

#read in sites
posfile='/Users/student/Documents/Algorithms/Alg_final_project/FinalProject/data/rap1-lieb-positives.txt'
negfile='/Users/student/Documents/Algorithms/Alg_final_project/FinalProject/data/yeast-upstream-1k-negative.fa'
testfile='/Users/student/Documents/Algorithms/Alg_final_project/FinalProject/data/rap1-lieb-test.txt'
poslist=util.read_pos(posfile)
posreversecomp=[]
for i in poslist:
    posreversecomp.append(util.reverse_complement(i))

poslist=poslist+posreversecomp
poslist=util.seq_encode(poslist)

neglist=util.read_fasta(negfile)
negreversecomp=[]
for i in neglist:
    negreversecomp.append(util.reverse_complement(i))
neglist=neglist+negreversecomp

#print('negs',neglist[:10])
print('neg',len(neglist))
print('pos',len(poslist))
shortneg=[]
for i in neglist:
    #TODO adapt so this is a random slice of the negative
    num=random.randint(0,len(i)-18)    
    shortneg.append(str(i[num:num+17]))


#encode neg
#neglist=util.seq_encode(shortneg)

neglist=shortneg
neglist=util.seq_encode(neglist)
#print('neglist',neglist[:10])
#sample x random seqs from positive and x from the negative to make sample
#keep track of which samples come from each group
#start with small training set

#sampleInputs,labels,samplelength=nn.shufflePosNegs(poslist,neglist)

#trainednet,final,mincost=nn.trainNet(sampleInputs, labels, samplelength,1000)

nn.CVtrainNet(poslist,neglist,1000,5)

'''
print('final cost',mincost)
print('final',final)
print('labels', labels)
print('rounded',final.round(decimals=2))
'''
#testlist=util.read_pos(testfile)
#print('test',testlist)
