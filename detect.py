import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

ID_sets = []
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
path='C:\Users\Mypc\Desktop\Project\images'
imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
for imagePath in imagePaths:
    tmp1 = 0
    tmp2 = 0
    for i in range(len(imagePath)-1,0,-1):
        if imagePath[i]=='m':
            tmp1=i
            break
    for i in range(len(imagePath) - 1 , 0 , -1):
	if imagePath[i]=='.':
	    tmp2=i
	    break
    ID=int(imagePath[tmp1+1:tmp2])
    ID_sets.append(ID)
ID=raw_input('enter your id:')
ID =int(ID)
while ID in ID_sets:
        ID=int(raw_input("ID Already Exists Enter Another Id:"))
        
img = cv2.imread('t11.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.22, 5)


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    ID=ID+1
    cv2.imwrite("images/m"+str(ID)+".jpg",gray[y:y+h,x:x+w])
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
   

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
