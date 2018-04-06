import numpy as np
import cv2,os
from matplotlib import pyplot as plt
from PIL import Image
import pickle
rec=cv2.createLBPHFaceRecognizer()
rec.load('trainner/trainner.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
img = cv2.imread('t11.jpg')
Id=0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0, 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.22, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    Id,conf=rec.predict(gray[y:y+h,x:x+w])
    cv2.cv.PutText(cv2.cv.fromarray(img),str(Id),(x,y+h),font,255)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

