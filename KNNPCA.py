
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
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import timeit
from sklearn import svm, f1_score


STANDARD_SIZE = (300, 167)
def img_to_matrix(filename, folder, verbose=False):
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


# First we are going to read the filenames in both directories
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
    #print image
    img = img_to_matrix(image,"1099_misc")
    img = flatten_image(img)
    data1.append(img)

for image in label2:
    #print image
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

""" tr is the size of training set and rest will be test set"""
tr= np.ceil(len(data)*.8)
te= len(data)- tr

Xtr= data[0:tr]
Ytr= labels[0:tr]
Xtes= data[tr:]
Ytes= labels[tr:]



scores= []
scores_std= []
K= range(1,202,20)



"""" K-NN with cross validation""""

for k in K:
    kf = KFold(len(Xtr),n_folds=5, shuffle=True)
    F1score =[]
    for train, test in kf:
        pca = decomposition.PCA(n_components=2)
        XTRAIN = pca.fit_transform(Xtr[train])

        clf = KNeighborsClassifier(k)
        clf.fit(XTRAIN,Ytr[train])
        #clf.fit(Xtr[train],np.zeros(len(Xtr[train])))
        XTEST = pca.transform(Xtr[test])
        Yhat = clf.predict(XTEST)
        #Error.append(1- abs(Ytr[test]-Yhat).mean())
        F1score = f1_score(Ytr[test], Yhat, average='weighted')
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
plt.ylabel('CV score')
plt.xlabel('Parameter k')
plt.ylim(0, 1.1)
plt.show()

nnst= timeit.default_timer()
fpca = decomposition.PCA(n_components=2)
XTR = fpca.fit_transform(Xtr)
XTES=  fpca.transform(Xtes)


clf = KNeighborsClassifier(80)
clf.fit(XTR,Ytr)
Yhat= clf.predict(XTES)
nnend = timeit.default_timer()

error= 1-float(sum(abs(Yhat-Ytes)))/float(len(Xtes))


"""" nnend - nnst is the time the classification  process"""
print error , nnend-nnst

##
##SVM = svm.SVC(kernel='rbf')
##SVM.C= 1
##SVM.fit(XTR,Ytr)
###Yhat= clf.predict(Xtes)
