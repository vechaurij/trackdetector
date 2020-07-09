import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from contourUtil import *

def findImageContours(imageFile):

    # Set image size
    imageSize = (800, 600)

    # Set the kernel size for image blur
    ksize = (5, 5) 

    img = cv2.imread(imageFile, cv2.IMREAD_UNCHANGED)

    # resize image
    imgResized = cv2.resize(img, imageSize)

    #convert img to grey
    imgGrey = cv2.cvtColor(imgResized,cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.blur(imgGrey,ksize)

    #set a thresh
    thresh = 80

    #get threshold image
    ret,imgThresh = cv2.threshold(imgBlur, thresh, 255, cv2.THRESH_BINARY)

    cv2.imshow("Threshold",imgThresh)
    cv2.waitKey()

    #find contours
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contourNum = np.size(contours)

    print("Contours found: " + str(contourNum))

    orderedContoursIndexes = getContoursOrdered(contours)

    #create an empty image for contours
    imgContours = np.zeros(imgResized.shape)

    for x in range(contourNum):

        # draw the contours on the empty image
        cv2.drawContours(imgContours, contours, orderedContoursIndexes[x], (0,255,0), 3)

        #save image
        cv2.imshow("contoursFound",imgContours)

        print("Contour: " + str(orderedContoursIndexes[x]) + " " + str(np.shape(contours[orderedContoursIndexes[x]])))
        cv2.waitKey()
        

    cv2.waitKey()


# Parse argumetns from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
                help = "Path to the input image filename")

args = vars(ap.parse_args())

findImageContours(args["image"])
