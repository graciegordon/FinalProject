from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

##This script will visualize a 3D plot

#read in file
with open('/Users/student/Documents/Algorithms/Alg_final_project/auc51H.txt','r') as f:
    alpha=[]
    lam=[]
    auc=[]
    for line in f:
        curline=line.split()
        #print(curline)
        alpha.append(float(curline[0]))
        lam.append(float(curline[1]))
        auc.append(float(curline[2]))

    #plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(alpha,lam,auc,c='b',marker='o')
    ax.set_xlabel('alpha')
    ax.set_ylabel('lambda')
    ax.set_zlabel('AUC')

    plt.title('51 Hidden Units')
    plt.show()


