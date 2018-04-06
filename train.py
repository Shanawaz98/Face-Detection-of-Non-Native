import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
path='C:\Users\Mypc\Desktop\Project\images'
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
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
        print ID
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("trainner",faceNp)
        cv2.waitKey(10)
    return IDs,faces

IDs,faces = getImagesWithID(path)
recognizer.train(faces, np.array(IDs))
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()
