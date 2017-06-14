

""" Code for Intuit project demonstration
         By: Amir Salimi"""

###  
import os
import numpy as np
import scipy
import sklearn
from PIL import Image
import random
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import decomposition
import timeit
import pylab as pl



""" Here we choose the standard size to our image"""
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
#tr=20
#tr=100
te= len(data)- tr

Xtr= data[0:tr]
Ytr= labels[0:tr]
Xtes= data[tr:]
Ytes= labels[tr:]



scores= []
scores_std= []



##
##
##
##clf = svm.SVC(kernel='linear')
##clf.C= 1
##clf.fit(Xtr,Ytr)
##Yhat= clf.predict(Xtes)
##
##error= 1-float(sum(abs(Yhat-Ytes)))/float(len(Xtes))
##




"""" Here we run PCA on the training set and
        then project test data on the same basis"""

pca = decomposition.PCA(n_components=2)
XTR = pca.fit_transform(Xtr)
XTES= pca.transform(Xtes)

knn = KNeighborsClassifier(80)
knn.fit(XTR,Ytr)
Yhat= knn.predict(XTES)

X=np.concatenate((XTR,XTES))
Y= np.concatenate((Ytr,Ytes))
# h is the step size here
h = 30
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(4, 3))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)

# Plot also the training points
pl.scatter(X[:,0], X[:,1],c=Y )


pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())

pl.show()



