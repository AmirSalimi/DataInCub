import math
import numpy as np
import scipy
import sklearn
from PIL import Image
from sklearn import svm
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#from sklearn.decomposition import RandomizedPCA
from sklearn import decomposition
from sklearn.metrics import f1_score
import timeit
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import axes3d
from sklearn.ensemble import RandomForestClassifier


"""This is number of important feature, for feature selection"""
impcount = 25


##################################################
### Reading the file and transforming to np.array
filename = "/Users/amir/Documents/DataProjects/Tempus/DScasestudy.txt"
f = open(filename)

datals =  []
for line in f:
    if str.isdigit(line[0]):
        row= [int(i) for i in line.strip().split()]
        datals.append(row)
data = np.array(datals)


##################
# I use %80 of data for training and %20 for testing
####################
tr = int(.8*len(data))
np.random.shuffle(data)
Xtr = data[:tr,1:]
Ytr = data[:tr,0]
Xtes = data[tr:,1:]
Ytes = data[tr:,0]





scores= []
scores_std= []
K= range(50,130,5)



scores= []
scores_std= []
K= range(1,10)

ntree = 100

"""" K-NN with cross validation"""
for k in K:
    kf = KFold(len(Xtr),n_folds=5, shuffle=True)
    Error =[]
    F1score =[]
    for train, test in kf:
        #clf = svm.SVC(kernel='linear', class_weight={1: 4})
        clf = RandomForestClassifier(n_estimators= ntree, min_samples_leaf=k ,n_jobs=-1, oob_score= True)
        clf.fit(Xtr[train],Ytr[train])

        Yhat = clf.predict(Xtr[test])
        print sum(abs(Ytr[test]-Yhat))
        Error.append(1- abs(Ytr[test]-Yhat).mean())
        F1score.append(f1_score(Ytr[test], Yhat, average='weighted'))
    scores.append(np.mean(F1score))
    scores_std.append(np.std(F1score))
print scores, scores_std
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(K, scores)
plt.plot(K, np.array(scores) + np.array(scores_std), 'b--')
plt.plot(K, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('F1 score')
plt.xlabel('min_samples_leaf')
plt.ylim(0, 1.1)
plt.title('N-tree= %d' %ntree)
plt.show()
nnst= timeit.default_timer()


"""" Final model goes here"""
clf = RandomForestClassifier(n_estimators= 100, min_samples_leaf=1 ,n_jobs=-1, oob_score= True)
clf.fit(Xtr,Ytr)
Yhat= clf.predict(Xtes)
nnend = timeit.default_timer()

error= 1-float(sum(abs(Yhat-Ytes)))/float(len(Xtes))
fscore= f1_score(Ytes, Yhat, average='weighted')
print "error:",error , "running time:",nnend-nnst, "fscore: ", fscore



###Plotting the importance of the features
importance = clf.feature_importances_
irange = np.arange(len(importance))

tups = zip(importance,irange)
mylist = sorted(tups, key=lambda x:-x[0])
objects = tuple([t[1] for t in mylist[:impcount]])
y_pos = np.arange(len(objects))
performance = [t[0] for t in mylist[:impcount]]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Feature relative importance')
plt.xlabel('Feature V*')
plt.title('N-tree= %d' %ntree)
 
plt.show()
                
