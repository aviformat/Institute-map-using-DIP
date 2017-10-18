import cv2
from random import randint
import numpy as np
import os
from sklearn import svm
import csv

hog = cv2.HOGDescriptor()
#im = cv2.imread("imagelnm.jpg")
#im=cv2.resize(im,(100,50))
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
        img_scaled = cv2.resize(img, (250, 200), interpolation=cv2.INTER_AREA)
        h = hog.compute(img_scaled)
        #print type(h)
        X1.append(h)
        #print X1
        y1.append(i)
    loc.append(x)
    i=i+1




print type(X1)
X1=np.hstack(X1)

X1=X1.transpose()

with open('matrix.txt', 'wb') as csvfile:
    matrixwriter = csv.writer(csvfile, delimiter=' ')
    for row in X1:
        matrixwriter.writerow(row)


with open('ylist.txt', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(y1)

with open('loclist.txt', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(loc)

print X1.shape,y1






clf=svm.LinearSVC()
clf.fit(X1,y1)


x2=cv2.imread("1.jpg")
img_scaled = cv2.resize(x2, (250, 200), interpolation=cv2.INTER_AREA)
h = hog.compute(img_scaled)

x2=np.hstack(h)

x2=x2.transpose()

ans=clf.predict(x2)

print loc[ans[0]]

