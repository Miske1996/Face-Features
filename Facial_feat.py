import numpy as np 
import cv2
from fastai.vision import *
from fastai.vision.image import pil2tensor , Image
from pathlib import Path
import glob
learn = load_learner('./','facial_features.pkl')
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
list_images = glob.glob('test_faces/*.jpeg')
# cp = cv2.VideoCapture(0)
for i in list_images: 
    # _,frame = cp.read()
    frame = cv2.imread(i)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    coords = fc.detectMultiScale(gray,1.1,5)
    max = 0
    max_id = 0
    for i in range(len(coords)):
        _, _, w_i, h_i = coords[i]
        if w_i*h_i > max:
            max_id = i
            max = w_i*h_i
        else:
            pass           
    x, y, w, h = coords[max_id]
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255))
    W,H,_ = frame.shape   
    fr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    prediction = str(learn.predict(Image(pil2tensor(fr,np.float32).div_(255)))[0]).split(';')         
    label = (" ".join(prediction)
            if "Male" in prediction
            else "Female " + " ".join(prediction))
    label_list = label.split(' ')
    size = (450, 450)
    frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
    for x,y,w,h in coords:
        for i in range(1,len(label_list)): 
                if label_list[i-1] == 'No_Beard':
                    del label_list[i-1]    
                cv2.putText(frame, label_list[i-1],(0,20*i),cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0, 255), 2 )  
    cv2.imshow('Face Features',frame)
    k = cv2.waitKey(6000)
    if k == 27:
        quit()
# cp.release
cv2.destroyAllWindows()