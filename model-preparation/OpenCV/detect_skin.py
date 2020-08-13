import cv2
import numpy as np
import pickle

cam = cv2.VideoCapture(0)
hist = None

while True:
    _,frame = cam.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame,(432,120),(575,365),(0,0,255),2)

    roi = frame[120:365,432:575]
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    cv2.imshow('roi',hsv)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(5)

    if k == ord('s'):
        hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
        break

cam.release()
cv2.destroyAllWindows()

with open("skin","wb") as file:
    pickle.dump(hist,file)
