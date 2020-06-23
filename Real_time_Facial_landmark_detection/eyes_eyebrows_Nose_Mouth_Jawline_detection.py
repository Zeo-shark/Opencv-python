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
#detector dlib's face detector(HOG_based)
#facial landmark predictor
detector= dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shape_predictor"])

#load the input image and resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image= imutils.resize(image, width=500)
gr= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect the faces in the gray
rects= detector(gr ,1)

#installing HOG based detector loading the facial landmark prediction
#detecting faces in the input image

for (i,rect) in enumerate(rects):
    #convert the landmark (x,y) coordinates in numpy array
    shape= predictor(gr, rect)
    shape= face_utils.shape_to_np(shape)# numpy array
    #loop over the face parts
    for (name, (i, j)) in face_utils.FACIAL_LANDMARK_IDXS.items():
        clone= image.copy()
        cv2.putText(clone, name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 2)

        for (x,y) in shape[i:j]:
            cv2.circle(clone, (x,y), 1,(0,0,255),-1)
        #extract the ROI of the face region
        (x,y,w,h)=cv2.boundingRect(np.array([shape[i:j]]))
        roi= image[y:y+h,x:x+w]
        roi= imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        cv2.imshow("ROI",roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)

    #visualize all facial landmark with a transparent overlay
    output= face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
#cv2.boundingRects(np.array([shape[i:j]))




