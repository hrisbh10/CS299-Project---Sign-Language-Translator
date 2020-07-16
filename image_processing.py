import pickle
import cv2
import numpy as np

def createMask(frame):
    p_in = open('skin','rb')
    hist = pickle.load(p_in)
    p_in.close()

    target = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',target)

    dst = cv2.calcBackProject([target],[0,1],hist,[0,180,4,256],1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE(12,12))
    cv2.filter2D(dst,-1,kernel1,dst)
    dst = cv2.GaussianBlur(dst,(15,15),0)
    dst = cv2.medianBlur(dst,15)
    ret,thresh = cv2.threshold(dst,35,255,cv2.THRESH_BINARY)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)

    return mask

cam = cv2.VideoCapture(0)

while True:
    _,frame = cam.read()
    frame = cv2.flip(frame,1)
    
    mask = createMask(frame)

    cv2.imshow('mask',mask)

    res = cv2.bitwise_and(frame,frame,mask=mask)
    canny = cv2.Canny(res,60,120)
    womask = cv2.Canny(frame,60,120)
    cv2.imshow('res',res)
    cv2.imshow('womask',womask)
    cv2.imshow('edges',canny)

    k = cv2.waitKey(5)
    if k == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
