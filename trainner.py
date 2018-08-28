import numpy as np
import cv2,os
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml');


def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePaths in imagePaths:
        pilImage=Image.open(imagePaths).convert('L')
        imageNp=np.array(pilImage, 'uint8')
        Id=int(os.path.split(imagePaths)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])  
            Ids.apend(Id)
        return faceSamples,Ids

faces,Ids=getImagesAndLabels('/home/aanchal')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')             


