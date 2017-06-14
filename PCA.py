

""" Code for Intuit project demonstration
         By: Amir Salimi"""


import os
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



# First we are going to read the filenames in both folders
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
    img2 = img_to_matrix(image,"other")
    img2 = flatten_image(img2)
    data2.append(img2)


###############  Partitioning into Training and Test data
    
data1 = np.array(data1)
idx1= range(len(data1))
random.shuffle(idx1)
data1= data1[idx1]


data2 = np.array(data2)
idx2= range(len(data2))
random.shuffle(idx2)
data2= data2[idx2]


data= np.concatenate((data1,data2))
labels= np.concatenate((np.ones(len(data1)),np.zeros(len(data2))))
##labels= labels[idx]
tr1= len(data1)

pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(data)
plt.scatter(X[:tr1,0],X[:tr1,1],color='red')
plt.scatter(X[tr1:,0],X[tr1:,1],color='blue')
plt.legend()
plt.show()
