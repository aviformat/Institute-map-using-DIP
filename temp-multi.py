import cv2
import numpy as np
from random import randint


img=cv2.imread('image.jpg',0)
w,h=img.shape

x=randint(0, w-100)
y=randint(0,h-100)
img1=img[x:x+100,y:y+100]
while(1):
    cv2.imshow("img1",img1)
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()
