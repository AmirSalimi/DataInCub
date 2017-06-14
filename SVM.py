



""" Code for Intuit project demonstration
         By: Amir Salimi"""


import os
import numpy as np
import scipy
import sklearn
#from scipy.misc import imread, imsave, imresize
from PIL import Image
import random
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import timeit




""" Here we choose the standard size to our image"""

STANDARD_SIZE = (300, 167)
def img_to_matrix(filename, folder):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open("/Users/amir/Data Science Craft Demo/ds_craft/"+folder+"/"+filename)
    img = img.convert('L')
    img = img.resize(STANDARD_SIZE)
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it 
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]




""" First we are going to read the filenames in both directories"""
label1, label2 = [], []
for file in os.listdir("/Users/amir/Data Science Craft Demo/ds_craft/1099_misc"):
    if file.endswith(".jpg"):
        label1.append(file)
for file in os.listdir("/Users/amir/Data Science Craft Demo/ds_craft/other"):
    if file.endswith(".jpg"):
        label2.append(file)


l1 = len(label1)
l2 = len(label2)
print l1, l2


data1 = []
data2 = []
for image in label1:
    img = img_to_matrix(image,"1099_misc")
    img = flatten_image(img)
    data1.append(img)

for image in label2:
    img = img_to_matrix(image,"other")
    img = flatten_image(img)
    data2.append(img)


###############  Partitioning into Training and Test data
    
data1 = np.array(data1)
data2 = np.array(data2)
data= np.concatenate((data1,data2))
labels= np.concatenate((np.ones(len(data1)),np.zeros(len(data2))))
idx= range(len(data))
random.shuffle(idx)
data= data[idx]
labels= labels[idx]


tr= np.ceil(len(data)*.8)

te= len(data)- tr

Xtr= data[0:tr]
Ytr= labels[0:tr]
Xtes= data[tr:]
Ytes= labels[tr:]



scores= []
scores_std= []
K= range(1,100,10)
#############  SVM Goes here with cross validation curve
for k in K:
    clf = svm.SVC(kernel='linear')
    clf.C = k
    kf = KFold(len(Xtr),n_folds=5, shuffle=True)
    Xtr, Ytr= shuffle(Xtr, Ytr)


    Error =[]
    for train, test in kf:
        clf = svm.SVC(kernel='linear', class_weight={1: 2})
        clf.C = k
        clf.fit(Xtr[train],Ytr[train])

        Yhat = clf.predict(Xtr[test])
        print sum(abs(Ytr[test]-Yhat))
        Error.append(1- abs(Ytr[test]-Yhat).mean())
    scores.append(np.mean(Error))
    scores_std.append(np.std(Error))
print scores, scores_std
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(K, scores)
plt.plot(K, np.array(scores) + np.array(scores_std), 'b--')
plt.plot(K, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()
nnst= timeit.default_timer()


"""" Final model goes here"""
clf = svm.SVC(kernel='linear')
clf.C= 1
clf.fit(Xtr,Ytr)
Yhat= clf.predict(Xtes)
nnend = timeit.default_timer()

error= 1-float(sum(abs(Yhat-Ytes)))/float(len(Xtes))
print "error:",error , "running time:",nnend-nnst
