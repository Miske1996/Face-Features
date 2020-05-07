import numpy as np 
import cv2
from fastai.vision import *
from fastai.vision.image import pil2tensor , Image
from pathlib import Path
learn = load_learner('./','facial_features.pkl')
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cp = cv2.VideoCapture(0)
while True : 
    # _,frame = cp.read()
    frame = cv2.imread('unknown_faces/emilia5.jpg')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    coords = fc.detectMultiScale(gray,1.1,5,minSize = (30,30))
    for x,y,w,h in coords:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255))
        W,H,_ = frame.shape
        x1, x2 = (max(0, x - int(w * 0.35)), min(x + int(1.35 * w), W))
        y1, y2 = (max(0, y - int(0.35 * h)), min(y + int(1.35 * h), H))       
        img_cp = frame[y1:y2,x1:x2].copy()
        img_cp1 = cv2.cvtColor(img_cp,cv2.COLOR_BGR2RGB)
        prediction = str(learn.predict(Image(pil2tensor(img_cp1,np.float32).div_(255)))[0]).split(';')         
        label = (" ".join(prediction)
                if "Male" in prediction
                else "Female " + " ".join(prediction))
        label_list = label.split(' ')
        for i in range(1,len(label_list)): 
            if label_list[i-1] == 'No_Beard':
                del label_list[i-1]    
            cv2.putText(frame, label_list[i-1],(x,y-14*i),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2 )
    cv2.imshow('Face Features',frame)
    k = cv2.waitKey(1)
    if k == 27:
        quit()
cap.release()
cv2.destroyAllWindows()