import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from kneebow.rotor import Rotor
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
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contourNum = np.size(contours)

    print("Contours found: " + str(contourNum))

    contours = removeImagePerimeterContour(imgThresh,contours)

    contourNum = np.size(contours)

    print("Contours found: " + str(contourNum))

    contoursIndexesAndAreasOrderedByArea = getContoursIndexesAndAreaOrderedByArea(contours)
    contoursIndexesAndSizesOrderedBySize = getContoursIndexesAndSizeOrderedBySize(contours)

    print("Contours with indexes and areas: " + str(contoursIndexesAndAreasOrderedByArea))
    print("Contours with indexes and sizes: " + str(contoursIndexesAndSizesOrderedBySize))
    rotor = Rotor()
    rotor.fit_rotate(contoursIndexesAndAreasOrderedByArea[:,1:3])
    elbowIndexAreas = rotor.get_elbow_index()

    rotor.fit_rotate(contoursIndexesAndSizesOrderedBySize[:,1:3])
    elbowIndexSizes = rotor.get_elbow_index()

    print("Elbow Index for areas: " + str(elbowIndexAreas))
    print("Elbow Index for sizes: " + str(elbowIndexSizes))

    highestElbow = elbowIndexAreas
    if (elbowIndexSizes > elbowIndexAreas):
        highestElbow = elbowIndexSizes

    #create an empty image for contours
    imgContours = np.zeros(imgResized.shape)

    # biggestGap = findBiggestGap(orderedContoursAreasByArea)

    # print("Biggest Gap: " + str(biggestGap))

    for x in range(contourNum):
        
        print("Contour: " + str(contoursIndexesAndAreasOrderedByArea[x,1]) + " Area: " + str(contoursIndexesAndAreasOrderedByArea[x,2]) + "Size: " + str(contoursIndexesAndSizesOrderedBySize[x,2]))

        if (x > highestElbow):
            # draw the contours on the empty image
            cv2.drawContours(imgContours, contours, contoursIndexesAndAreasOrderedByArea[x,1], (0,255,0), 3)

            #save image
            cv2.imshow("contoursFound",imgContours)

            cv2.waitKey()


# Parse argumetns from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
                help = "Path to the input image filename")

args = vars(ap.parse_args())

findImageContours(args["image"])
