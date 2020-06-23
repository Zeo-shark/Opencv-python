from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FileVideoStream
import  numpy as np
import argparse
from imutils import face_utils
import imutils
import  time
import dlib
import cv2

def eye_aspect_ratio(eye):
    #compute the euclidean dist between two sets
    #vertical eye landmark(x,y)
    A= dist.euclidean(eye[1],eye[5])
    B= dist.euclidean(eye[2],eye[4])
    c= dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*c)

    return ear
ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor", required=True,help="path to the facial landmark")
ap.add_argument("-v","--video",type=str,default="",help="path to the video file")
args=vars(ap.parse_args())

EYE_AR_THRESH=0.3
EYE_AR_CONSEC_FRAMES= 3

COUNTER=0
TOTAL=0
detector= dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor(args["shape-predictor"])

#histogram of Oriented gradients + linear SVM
(lStart,lEnd)= face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart,rEnd)= face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print("Starting Video Stream Thread......")
vs=FileVideoStream(args["video"]).start()
fileStream=True

time.sleep(2.0)
while True:
    if fileStream and not vs.more():
        break
    frame= vs.read()
    frame= imutils.resize(frame, width=450)
    gr= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects= detector(gr,0)
    #loop over the face detections
    for rect in rects:
        shape= predictor(gr, rect)
        shape= face_utils.shape_to_np(shape)

        leftEye= shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR= eye_aspect_ratio(leftEye)
        rightEAR= eye_aspect_ratio(rightEye)
        ear=(leftEAR+rightEAR)/2.0
        if ear < EYE_AR_THRESH:
            COUNTER+=1

        #otherwise the eye aspect ratio is not below the blink threshold
        else:
            if COUNTER>= EYE_AR_CONSEC_FRAMES:
                TOTAL+=1
            #reset the eye frame counter
            COUNTER=0
            cv2.putText(frame, "Blinks: {}".format(TOTAL),(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            cv2.putText(frame,"EAR: {}".format(ear),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255), 2)
        cv2.imshow("Frame",frame)
        key= cv2.waitKey(1)& 0xFF

        if key==ord('q'):
            break
cv2.destroyAllWindows()
vs.stop()
