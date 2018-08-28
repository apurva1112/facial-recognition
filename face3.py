

import numpy as np
import cv2

recognizer=cv2.face.LBPHFaceRecognizer_create()
#recognizer.load('trainner/trainner.yml') 
cascadePath="haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while(True):
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        Id,conf=recognizer.predict(gray[y:y+h,x:x+h])
        if(conf<50):
        	if(Id==1):
        		Id="AANCHAL"
        	elif(Id==2):
        		Id="MUSKAN"
        	else:
        		Id="unknown"
        	cv2.cv.PutText(cv2.cv.fromarray(im), str(Id), (x,y+h),font, 255)	
    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()