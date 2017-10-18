import cv2
from random import randint
import numpy as np
import os
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install

from itertools import repeat
d = [[] for i in repeat(None, 70)]
i=0
#print d
from sklearn import svm
a=[]
locations=[]
X1=[]
y1=[]
loc=[]
mat=np.matrix(X1)
for x in os.listdir("images"):
    locations.append(x)

for x in os.listdir("images"):
    for y in os.listdir("images/" + x):
        im=cv2.imread("images/"+x+"/"+y)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img_scaled = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
        radius = 32
        # Number of points to be considered as neighbourers
        no_points = 32 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(img_scaled, no_points, radius, method='uniform')
        # Calculate the histogram
        ab = itemfreq(lbp.ravel())
        # Normalize the histogram
        hist = ab[:, 1] / sum(ab[:, 1])
        #a.append(hist)
        #print a

        hist = np.asarray(hist)
        print len(hist)
        #print len(hist)

        #X1=np.stack((X1,hist))
        #hist=np.asarray(hist)
        X1.append(hist)
        y1.append(i)
    i=i+1
    loc.append(x)

        #print hist.shape
        #X1.append(a)
print len(X1)
print X1
# X1=np.reshape(X1,(70,130))
# X1 = np.array(X1)
# #print X1
# #X1=X1.transpose()
# print X1
#
# for i in range(70):
#     for j in range(130):
#         X1[i][j]=float(X1[i][j])
#
# #print X1
#
# #X1=X1.vstack(X1)
# #X1=X1.tolist()
# #print X1.shape
# #X1=np.hstack(X1)
# #print X1
#
#
# clf=svm.LinearSVC()
# clf.fit(X1,y1)
#
#
# im=cv2.imread("4.jpg")
# img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# img_scaled = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
# radius = 4
# # Number of points to be considered as neighbourers
# no_points = 32 * radius
# # Uniform LBP is used
# lbp = local_binary_pattern(img_scaled, no_points, radius, method='uniform')
# # Calculate the histogram
# ab = itemfreq(lbp.ravel())
# # Normalize the histogram
# hist = ab[:, 1] / sum(ab[:, 1])
# hist = np.asarray(hist)
# ans=clf.predict(hist)
# print hist
# print ans
# print loc
# print loc[ans[0]]
#
# #
