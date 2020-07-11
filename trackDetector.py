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


    #isContourClosed(contours[0])

    orderedContoursIndexesByArea, orderedContoursAreasByArea = getContoursOrderedByArea(contours)
    orderedContoursIndexesBySize, orderedContoursAreasBySize = getContoursOrderedBySize(contours)
    orderedContoursByArea = sorted(contours, key = cv2.contourArea, reverse = True)[1:2]

    contoursIndexesAndAreasOrderedByArea = getContoursIndexesAndAreaOrderedByArea(contours)

    # Z = clusterContoursKmeans(contours)
    # Z = clusterContoursKmeans2(contours)
    # Z = clusterContoursKmeans3(contours)

    #slopes = getSlopes(orderedContoursAreasByArea)

    contoursIndexesAndAreasOrderedByArea = getContoursIndexesAndAreaOrderedByArea(contours)

    print("Contours with indexes and areas: " + str(contoursIndexesAndAreasOrderedByArea))
    rotor = Rotor()
    rotor.fit_rotate(contoursIndexesAndAreasOrderedByArea[:,1:3])
    elbow_idx = rotor.get_elbow_index()
    print("Elbow Index: " + str(elbow_idx))
    rotor.plot_elbow()
    cv2.waitKey()


    #create an empty image for contours
    imgContours = np.zeros(imgResized.shape)

    # biggestGap = findBiggestGap(orderedContoursAreasByArea)

    # print("Biggest Gap: " + str(biggestGap))

    for x in range(contourNum):
        
        print("Contour: " + str(orderedContoursIndexesByArea[x]) + " " + str(np.shape(contours[orderedContoursIndexesByArea[x]])) + "Area: " + str(cv2.contourArea(contours[orderedContoursIndexesByArea[x]])))

        if (x > elbow_idx):
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
