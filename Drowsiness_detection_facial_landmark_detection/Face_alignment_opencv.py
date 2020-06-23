from imutils.face_utils import FACIAL_LANDMARKS_IDXS
from imutils.face_utils import shape_to_np
import numpy as np
import cv2
class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35,0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        self.predictor=predictor
        self.desiredFaceWidth= desiredFaceWidth
        self.desiredLeftEye= desiredLeftEye
        self.desiredFaceHeight = desiredFaceHeight

        if desiredFaceHeight is None:
            self.desiredFaceHeight=self.desiredFaceWidth

    def align(self, image, gray,rect):
        shape= self.predictor(gray,rect)
        shape= shape_to_np(shape)
        (lStart,lEnd)=FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart,rEnd)=FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEyePts= shape[lStart:lEnd]
        rightEyePts=shape[rStart:rEnd]

        leftEyeCentre= leftEyePts.mean(axis=0).astype("int")
        rightEyeCentre=rightEyePts.mean(axis=0).astype("int")

        dY= rightEyeCentre[1]-leftEyeCentre[1]
        dX=rightEyeCentre[0]-rightEyeCentre[0]

        angle= np.degree(np.arctan(dY,dX))-180

        desiredRightEyeX= 1.0- self.desiredLeftEye[0]
        #determine the scale between the current and desired image

        dist= np.sqrt((dX**2)+(dY**2))
        desiredDist= (desiredRightEyeX- self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale= desiredDist/dist

        eyesCentre= ((leftEyeCentre[0]+ rightEyeCentre[0])//2,((leftEyeCentre[0]+ rightEyeCentre[0])//2))

        M= cv2.getRotationMatrix2D(eyesCentre, angle, scale)

        tX= self.desiredFaceWidth*0.5
        tY= self.desiredFaceHeight* self.desiredLeftEye[1]
        M[0,2]+=(tX- eyesCentre[0])
        M[1,2]+=(tY- eyesCentre[1])
        #apply the affine transformation
        (w, h)= (self.desiredFaceWidth, self.desiredFaceHeight)
        output= cv2.warpAffine( image, M,(w, h), flags=cv2.INTER_CUBIC)

        return output








