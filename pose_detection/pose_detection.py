import cv2
protoFile="pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile="pose/mpi/pose_iter_160000.caffemodel"

net=cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

frame= cv2.imread("single.jpg")

inWidth=368
inHeight=368

blob= cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth,inHeight),(0,0,0), swapRB=False, crop=False)

net.setInput(blob)
output= net.forward()

H=output.shape[2]
W=output.shape[2]
points=[]
for i in range(len()):
    #confidence map of the corresponding body part
    probMap=output[0,i, :, :]
    #Find the global maxima of the probMap
    minVal, prob , minLoc, point=cv2.minMaxLoc(probMap)
    #Scale the point to fit on the original image
    x=(inWidth*point[0])/W
    y=(inHeight*point[1])/H

    cv2.circle(frame,(int(x),int(y)), 15, (0,255,255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frame, "{}".format(i),(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,0,255), 3, lineType=cv2.LINE_AA)
    points.append((int(x),int(y)))

    points.append(None)
cv2.imshow("Output-Keypoints", frame)
cv2.destroyAllWindows()
