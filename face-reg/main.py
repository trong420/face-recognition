import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'image'
images = []
classnames = []
List = os.listdir(path)
for a in List:
    cur = cv2.imread(f'{path}/{a}')
    images.append(cur)
    classnames.append(os.path.splitext(a)[0])


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('Checkin.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{string}')

encodeListKnown = findEncodings(images)
print('Complete Encoding')
print('Start Webcam')
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex]
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 153), 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 0, 153), cv2.FILLED)
            cv2.putText(img, name, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


