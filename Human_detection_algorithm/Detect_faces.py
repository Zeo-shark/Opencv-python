import numpy as  np
import cv2
import argparse

#construct the argument parse and the parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
ap.add_argument("-p","--prototxt",required=True,help="path to prototxtfile")
ap.add_argument("-m","--model",required=True,help="Path to Caffe pretrained model")
ap.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probablity to weak detections")
args= vars(ap.parse_args())#parsing command line argument using argparse
#--image
#--prototxt
#--model pretrained caffe model
#optional confidence default threshold of 0.5
#create a blob from our image
print("[INFO] loading the model....")
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])
#load the input image and construct an input blob from image
#resizing and nornmalising
image=cv2.imread(args["image"])

(h,w)=image.shape[:2]
blob= cv2.dnn.blobFromImage(cv2.resize(image, (300,300)),1.0,(300,300),(104.0,177.0,123.0))

#pass the blob through the network
#detection andpredictions
print("[INFO] computing the detections.....")
net.setInput(blob)
detections=net.forward()

#loop over the detections
for i in range(0,detections.shape[:2]):
    #extract confidence associated with the
    confidence=detections[0, 0, i, 2]
    #filter the weak predictions by confidence>min confidence
    if confidence > args["confidence"]:
        #compute (x,y) bounding box for the object
        box=detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype("int")
        #draw bounding with confidence
        text="{:.2f}%".format(confidence*100)
        y=startY- 10 if startY -10 >10 else startY+10
        cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(image, text,(startX,y),cv2.FONT_HERSHEYSIMPLEX,0.45,(0,0,255),2)
    cv2.imshow("Output",image)
    cv2.waitKey(0)


