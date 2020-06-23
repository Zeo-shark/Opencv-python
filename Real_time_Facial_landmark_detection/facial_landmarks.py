from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor", required= True, help="path to facial landmark prediction")
ap.add_argument("-i", "--image", reqiured= True, help="Path to the input image")
args = vars(ap.parse_args())

detector= dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor(args["shape_predictor"])
#Histogram of oriented Gradients+ linear SVM

image= cv2.imread(args["image"])
image= cv2.resize(image, width=500)
gr= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects= detector(gr, 1)
#loop over the face detection
for (i,rect) in enumerate(rects):
    #detect facial landmarks convert to numpy (x,y) array
    shape =  predictor(gr, rect)
    shape= face_utils.shape_to_np(shape)

    (x,y,w,h)=face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0), 2)

    #show the face number
    cv2.putText(image,"Face #{}",format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)

    for (x,y) in shape:
        cv2.circle(image, (x,y),1,(0,255,0), -1)
cv2.imshow("Output", image)
cv2.waitKey(0)
