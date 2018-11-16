import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import pygame
import datetime
from time import sleep

lastTime = datetime.datetime.now()
sleep(4)
currentTime = datetime.datetime.now()
pygame.mixer.init()
pygame.mixer.set_num_channels(2)
sound = pygame.mixer.Channel(1)
siren = pygame.mixer.Sound("siren.wav")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = VideoStream(src=0).start()

while cap.stream.isOpened():
    frame = cap.read()
    frame = imutils.resize(frame, width=608)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        if sound.get_busy() == False:
            if (currentTime - lastTime).seconds > 4:
                LastTime = datetime.datetime.now()
                sound.play(siren)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        currentTime = datetime.datetime.now()

    cv2.imshow('frame', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.stop()
cap.stream.release()
cv2.destroyAllWindows()
