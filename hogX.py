import csv
import cv2
from random import randint
import numpy as np
import os
from sklearn import svm

X1 = []
y1=[]
loc=[]
hog = cv2.HOGDescriptor()
with open('matrix.txt', 'rb') as csvfile:
    matrixreader = csv.reader(csvfile, delimiter=' ')
    for row in matrixreader:
        X1.append(row)
print "done"
with open("ylist.txt", 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    y1 = list(reader)
    #print(my_list)
print "done"
with open("loclist.txt", 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    loc = list(reader)
    #print(my_list)
print "done"




clf=svm.LinearSVC()
clf.fit(X1,y1)

print "done"

x2=cv2.imread("1.jpg")
img_scaled = cv2.resize(x2, (250, 200), interpolation=cv2.INTER_AREA)
h = hog.compute(img_scaled)

x2=np.hstack(h)

x2=x2.transpose()

ans=clf.predict(x2)

print loc[ans[0]]
