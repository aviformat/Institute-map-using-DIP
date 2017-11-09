import os
import cv2
import csv
import pickle
import numpy as np
from sklearn import svm
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


print cv2.useOptimized()
e1=cv2.getTickCount()

hog = cv2.HOGDescriptor()
locations=[]
X1=[]
y1=[]
loc=[]
i=0
for x in os.listdir("images"):
    locations.append(x)

for x in os.listdir("images"):
    for y in os.listdir("images/" + x):
        img=cv2.imread("images/"+x+"/"+y)
        img_scaled = cv2.resize(img, (125, 125) , interpolation=cv2.INTER_AREA)
        h = hog.compute(img_scaled)
        X1.append(h)
        y1.append(i)
    loc.append(x)
    i=i+1


X1=np.hstack(X1)
X1=X1.transpose()




parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10],'gamma':[0,10]}

clf=svm.SVC()
clf = GridSearchCV(clf, parameters)
clf.fit(X1,y1)

with open('hogModel.pkl', 'wb') as fr:
    pickle.dump(clf, fr)

print("dumped\n")

with open('hogModel.pkl', 'r') as fr:
    clf = pickle.load(fr)

print("model loaded\n")


x2=cv2.imread("5.jpg")
img_scaled = cv2.resize(x2, (125, 125), interpolation=cv2.INTER_AREA)
h = hog.compute(img_scaled)

x2=np.hstack(h)

x2=x2.transpose()
x2=x2.reshape(1,-1)
ans=clf.predict(x2)
print ans
print loc[ans[0]]