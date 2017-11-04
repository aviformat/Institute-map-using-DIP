import cv2
from random import randint
import numpy as np
import os
from sklearn import svm
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


print cv2.useOptimized()
e1=cv2.getTickCount()

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
        img_scaled = cv2.resize(img, (125, 125) , interpolation=cv2.INTER_AREA)
        h = hog.compute(img_scaled)
        #print type(h)
        X1.append(h)
        #print X1
        y1.append(i)
        #print len(h)
    loc.append(x)
    i=i+1




print type(X1)
X1=np.hstack(X1)

X1=X1.transpose()

#plt.scatter((0,1),X1)

# with open('matrix.txt', 'wb') as csvfile:
#     matrixwriter = csv.writer(csvfile, delimiter=',')
#     for row in X1:
#         matrixwriter.writerow(row)
#
#
# with open('ylist.txt', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(y1)
#
# with open('loclist.txt', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(loc)
#
# print X1.shape,y1




parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10],'gamma':[0,10]}

clf=svm.SVC()
clf = GridSearchCV(clf, parameters)
print clf.fit(X1,y1)


# clf=svm.SVC()
# clf.fit(X1,y1)

# x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
# y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
# h = (x_max / x_min)/100
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#  np.arange(y_min, y_max, h))
#
# plt.subplot(1, 1, 1)
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
# plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.title('SVC with linear kernel')
# plt.show()
#



x2=cv2.imread("5.jpg")
img_scaled = cv2.resize(x2, (125, 125), interpolation=cv2.INTER_AREA)
h = hog.compute(img_scaled)

x2=np.hstack(h)

x2=x2.transpose()
print x2.shape
x2=x2.reshape(1,-1)
ans=clf.predict(x2)
print ans
print loc[ans[0]]

e2=cv2.getTickCount()
t=(e2-e1)/cv2.getTickFrequency()
print t