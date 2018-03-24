#This code will take train a neural network to detect RAP1 sites

import neural_network as nn
import utils as util
from random import shuffle
import numpy as np
import random

#read in sites
posfile='/Users/student/Documents/Algorithms/Alg_final_project/data/rap1-lieb-positives.txt'
negfile='/Users/student/Documents/Algorithms/Alg_final_project/data/yeast-upstream-1k-negative.fa'
testfile='/Users/student/Documents/Algorithms/Alg_final_project/data/rap1-lieb-test.txt'
poslist=util.read_pos(posfile)
finaltestlist=util.read_pos(testfile)
posreversecomp=[]
for i in poslist:
    posreversecomp.append(util.reverse_complement(i))

poslist=poslist+posreversecomp

neglist=util.read_fasta(negfile)
negreversecomp=[]
for i in neglist:
    negreversecomp.append(util.reverse_complement(i))
neglist=neglist+negreversecomp

for i in neglist:
    if i in set(poslist):
        neglist.remove(i)

#print('negs',neglist[:10])
print('neg',len(neglist))
print('pos',len(poslist))
shortneg=[]
for i in neglist:
    #TODO adapt so this is a random slice of the negative
    num=random.randint(0,len(i)-18)    
    shortneg.append(str(i[num:num+17]))


#encode pos and negs
poslist=util.seq_encode(poslist)
finaltest=util.seq_encode(finaltestlist)
neglist=shortneg
neglist=util.seq_encode(neglist)


#sample x random seqs from positive and x from the negative to make sample
#keep track of which samples come from each group
#start with small training set


#alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#lambdas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

afile=open('auc.txt','w')

alphas=[0.1]
lambdas=[0.5]
for a in alphas:
    for l in lambdas:
        
        auc=nn.CVtrainNet(poslist,neglist,finaltest,2000,3,a,l)
        
        afile.write(str(a)+'\t'+str(l)+'\t'+str(auc)+'\n')

afile.close()
