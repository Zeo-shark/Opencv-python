from __future__ import print_function
from keras.models import load_model
from keras.datasets import cifar10
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap= argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help= "path to the output file")
ap.add_argument("-t","--test-images",required=True,help= "pathto the directory of testing images")
ap.add_argument("-b","--batch-size",type = int, default= 32,help="size of mini batches passed to the network")
args=vars(ap.parse_args())

gtlabels= ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

model= load_model(args["model"])

print("[INFO] Sampling CIFAR10......")
(testData,testlabels)=cifar10.load_data()[1]
testData=testData.astype("float") /255.0
np.random.seed(42)
idxs= np.random.choice(testData.shape[0],size=(15,), replace=False)
(testData, testlabels)= (testData[idxs],testlabels[idxs])
testlabels= testlabels.flatten()

print("Predicting on testing data")
probs= model.predict(testData,batch_size=args["batch_size"])
predictions= probs.argmax(axis = 1)
for (i,prediction) in enumerate(predictions):
    image= testData[i].astype(np.float32)
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image= imutils.resize(image, width= 128, inter= cv2.INTER_CUBIC)

    print("Predicted : {}, actual: {}".format(gtlabels[prediction],gtlabels[testlabels[i]]))
    cv2.imshow("Image",image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
print("Testing on images not part of CIFAR-10")

for imagePath in paths.list_images(args["test_images"]):

    image= cv2.imread(imagePath)
    kerasimage=cv2.resize(image, (32,32))
    kerasimage= cv2.cvtColor(kerasimage,cv2.COLOR_BGR2RGB)
    kerasimage= np.array(kerasimage, dtype="float")/255.0

    kerasimage= kerasimage[np.newaxis,...]
    probs= model.predict(kerasimage,batch_size=args["batch_size"])
    predictions= probs.argmax(axis=1)[0]
    cv2.imshow("Image",image)
    cv2.waitKey(0)

