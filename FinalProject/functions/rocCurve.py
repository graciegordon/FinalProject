
import matplotlib.pyplot as plt
import string
import numpy as np
import sys

def GetRates(truthlabel, scores):
    #convert numpy array to list
    #truthlabel=truthlabel.tolist()
    #scores=scores.tolist()
    #print(truthlabel)
    #print(scores)
    tpr = []  # true positive rate
    fpr = []  # false positive rate
    totsamples=len(truthlabel)
    #print('tot',totsamples)
    numtruepos = truthlabel.count(1)
    numtrueneg = len(truthlabel) - numtruepos
    #print('truepos',numtruepos)
    #print('trueneg',numtrueneg)
    foundtruepos = 0.0
    foundfalsepos = 0.0
    thresholds1=np.arange(0,1.1,0.1)
    fprI=thresholds1
    #print(thresholds1)
    thresholds=thresholds1[::-1]

    #start where positives =1 only then soften the thresholds iteratively
    for thresh in thresholds:
        #print('threshold',thresh)
        foundtruepos=0
        foundfalsepos=0
        for truth,pred in zip(truthlabel,scores):
            #for this threshold these are positives
            
            if pred >= thresh:
                #print('true',pred,'thresh',thresh)
                cur=1.0
                #print('cur', cur,'pred',pred)
                if cur == truth:

                    #print('truepos')
                    foundtruepos+=1
                if cur != truth: 
                    foundfalsepos+=1
                    #print('falsepos')

        tpr.append(foundtruepos / float(numtruepos))
        fpr.append(foundfalsepos / float(numtrueneg))
    ''' 
    joint=[]
    for t,f in zip(tpr,fpr):
        joint.append((tpr,fpr))

    joint=sorted(joint, key=lambda x: x[0])

    tpr=[ x[0] for x in joint ]
    fpr=[ x[1] for x in joint ]
    '''
    #print(tpr)
    #print(fpr)
    return tpr, fpr

def DepictROCCurve(truthlabel, scores, linelabel, color, fname, randomline):

    plt.figure(figsize=(4, 4), dpi=80)

    SetupROCCurvePlot(plt)
    AddROCCurve(plt, truthlabel, scores, color, linelabel)
    SaveROCCurvePlot(plt, fname, randomline)

def SetupROCCurvePlot(plt):

    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("RAP1 Binding Sites", fontsize=14)

def AddROCCurve(plt, truthlabel, scores, color, linelabel):

    tpr, fpr = GetRates(truthlabel, scores)

    plt.plot(fpr, tpr, color=color, linewidth=5, label=linelabel)

def SaveROCCurvePlot(plt, fname, randomline=True):

    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(fname)

def Create_IDs(pos,neg):
    poslen=len(pos)
    neglen=len(neg)
    letters=list(string.ascii_lowercase)
    letters.extend([i+b for i in letters for b in letters])
    #print(letters)
    posID=letters[:poslen]
    negID=letters[poslen:poslen+neglen]
    scores=[]
    for i in range(poslen):
        scores.append((posID[i],pos[i]))
    for i in range(neglen):
        scores.append((negID[i],neg[i]))

    scores=sorted(scores,key=lambda x: x[1],reverse=True)
    return scores,posID


#truths=[0,1,0,1,1,0,1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,0]
#guess=[0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,0.997,0.0,0.0,0.001,1.0,1.0,0.975,1.0,0.987,0.0,0.0,1.0,0.999,1.0,0.898,0.0,1.0,0.0,0.0,0.0,0.1,1.0,0.0,0.345,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,0.0]

#DepictROCCurve(truths, guess, 'preds', 'b', 'testPlot.png', 'True')


