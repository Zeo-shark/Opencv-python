from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True, help="path to the facial landmark predictor")
ap.add_argument("-i","--images", required=True, help="path to the input image")
args= vars(ap.parse_args())

detector= dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor()
fa= FaceAligner(predictor, desiredFaceWidth= 256)

image= cv2 .imread(args["image"])
image= imutils.resize(image, width=800)
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image",image)
rects= detector(gray,2)

for rect in rects:
    (x,y,w,h)= rect_to_bb(rect)
    feceOrig= imutils.resize(image[y:y+h,x:x+w], width=256)
    faceAligned= fa.align(image, gray,rect)

    cv2.imshow("Original",faceOrig)
    cv2.imshow("Aligned",faceAligned)
    cv2.waitKey(0)


