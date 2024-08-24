from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import cvlib as cv
import numpy as np
import os
import pywhatkit
import json
from urllib.request import urlopen
import time
from datetime import datetime

url='http://ipinfo.io/json'
response=urlopen(url)
data=json.load(response)
city =data['city']
state =data['region']

model = load_model('gender_detection.keras')

webcam = cv2.VideoCapture(0)
classes = ['Male','Female']

while webcam.isOpened():
    status,frame = webcam.read()

    face,confidence = cv.detect_face(frame)

    no_men=0
    no_women=0
    for idx,f in enumerate(face):
        (startX,startY) = f[0],f[1]
        (endX,endY) = f[2],f[3]

        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)

        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10  or (face_crop.shape[1]) < 10:
            continue

        face_crop = cv2.resize(face_crop,(96,96))
        face_crop = face_crop.astype("float")/255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop,axis=0)

        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]

        if label=='Female':
            no_women+=1
        else:
            no_men+=1
        
        label = "{}: {:.2f}%".format(label,conf[idx]*100)
        
        Y = startY - 10 if startY - 10 > 10 else startY + 10




        cv2.putText(frame,label,(startX,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    print(f"Number of men: {no_men}")
    print(f"Number of women : {no_women}")
    print()
    print()
    print()
    print()
    if(no_women==1 and no_men>=4):
        t=datetime.now()
        h=t.hour
        m=t.minute+1
        if m >= 60:
            h += 1
            m -= 60
        pywhatkit.sendwhatmsg("+919600481496",f"Women surrounded by/spotted with {no_men} men at {city} in {state}",h,m)
    elif(no_women==1 and no_men==0):
        t=datetime.now()
        h=t.hour
        m=t.minute+1
        if m >= 60:
            h += 1
            m -= 60
        pywhatkit.sendwhatmsg("+919600481496",f"lone women spotted at {city} in {state}",h,m)



    cv2.imshow("gender detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()