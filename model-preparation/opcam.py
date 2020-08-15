import cv2
import numpy as np
from tensorflow import keras
import pickle
import os

BATCH_SIZE = 64
IMG_SIZE = 192

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
labels = []
with open('labels.txt','r') as fin:
	labels = fin.readlines()
labels = [x[:-1] for x in labels]

TEST_DIR = "asl_alphabet_test"
model = keras.models.load_model('mobilenetv2_192_kaggle_bm142.h5')

#hist = None
#with open('skin','rb') as p_in:
#    hist = pickle.load(p_in)


#def createMask(frame):
  
#    target = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#    #cv2.imshow('hsv',target)

#    dst = cv2.calcBackProject([target],[0,1],hist,[0,180,0,256],1)
#    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
#    cv2.filter2D(dst,-1,kernel1,dst)
#    dst = cv2.GaussianBlur(dst,(15,15),0)
#    dst = cv2.medianBlur(dst,15)
#    ret,thresh = cv2.threshold(dst,35,255,cv2.THRESH_BINARY)
#    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
#    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)

#    return mask

def keras_predict(img):
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img_arr = (np.array([img])/255.0)
    pred = model.predict(img_arr)
    mxp = np.argmax(pred[0])
    txt = '{:.2f} '.format(pred[0][mxp]*100)
    if pred[0][mxp]*100 < 80:
        txt += 'unclassed'
    else:
    	txt += labels[mxp]
    return txt

cam = cv2.VideoCapture(0)

while True:
    _,frame = cam.read()
    frame = cv2.flip(frame,1)
    #mask = createMask(frame)
    #res = cv2.bitwise_and(frame,frame,mask=mask)
    res = frame
    txt = keras_predict(res[82:392,333:600])
    cv2.rectangle(res,(333,82),(600,392),color=(25,233,4),thickness=3)
    cv2.putText(res,txt,(333,80),cv2.FONT_HERSHEY_SIMPLEX,1,color=(25,233,4),thickness=4)
    cv2.imshow('frame',res)
    
    k = cv2.waitKey(4) & 0xFF
   
    if k==27:
        break

cv2.destroyAllWindows()
cam.release()

