import cv2
from random import randint
import numpy as np
import os
import temp

locations=[]

input=cv2.imread("imagelnm.jpg",0)
for x in os.listdir("images"):
    locations.append(x)

for x in os.listdir("images"):
    for y in os.listdir("images/"+x):
        value=temp.check("images/"+x+"/"+y,input)
        print value,x
        #print "images/"+x+"/"+y
