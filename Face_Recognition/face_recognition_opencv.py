# 얼굴 탐지
import cv2, sys
import numpy as np


def faceDetect():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    try:
        # 이면 캠
        cap = cv2.VideoCapture(0)
    except:
        print('카메라 로딩 실패')
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (10, 10))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3, 4, 0)

        frame = cv2.resize(frame.copy(), (800, 700))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyWindow('frame')
            break;


faceDetect()