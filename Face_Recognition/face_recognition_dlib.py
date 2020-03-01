import cv2, dlib, sys
import numpy as np
import PIL as ImageGrab
import time

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_detector = dlib.get_frontal_face_detector()

def faceDetect():
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
            # landmark
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
            landmarks_display = landmarks[0:68]

            for idx, point in enumerate(landmarks_display):
                pos = (point[0, 0], point[0, 1])
                cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

        frame = cv2.flip(frame, 1, 0)
        # frame = cv2.resize(frame.copy(), (800,700))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyWindow('frame')
            break


faceDetect()