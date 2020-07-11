import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.cluster.vq import kmeans, vq

def getContoursOrderedBySize(contours):

    contoursNum = np.size(contours)

    contoursIndexesWithSize = np.empty(contoursNum*2,dtype='i').reshape(contoursNum,2)
 
    for i in range(contoursNum):
        contoursIndexesWithSize[i,0] = i
        contoursIndexesWithSize[i,1] = np.size(contours[i])
        #print("Contour: " + str(i) + " size: " + str(np.size(contours[i])))

    # Order the contoursIndexesWithSize array by it's second column
    contoursIndexesWithSizeReverseOrder = contoursIndexesWithSize[contoursIndexesWithSize[:,1].argsort()][::-1]

    return contoursIndexesWithSizeReverseOrder[:,0],contoursIndexesWithSizeReverseOrder[:,1]

def getContoursOrderedByArea(contours):

    contoursNum = np.size(contours)

    contoursIndexesWithArea = np.empty(contoursNum*2,dtype='i').reshape(contoursNum,2)
 
    for i in range(contoursNum):
        contoursIndexesWithArea[i,0] = i
        contoursIndexesWithArea[i,1] = cv2.contourArea(contours[i])

    # Order the contoursIndexesWithArea array by it's second column
    contoursIndexesWithAreaReverseOrder = contoursIndexesWithArea[contoursIndexesWithArea[:,1].argsort()][::-1]

    return contoursIndexesWithAreaReverseOrder[:,0],contoursIndexesWithAreaReverseOrder[:,1]

def getContoursIndexesAndAreaOrderedByArea(contours):

    contoursNum = np.size(contours)

    contoursIndexesWithArea = np.empty(contoursNum*3,dtype='i').reshape(contoursNum,3)
 
    for i in range(contoursNum):
        contoursIndexesWithArea[i,1] = i
        contoursIndexesWithArea[i,2] = cv2.contourArea(contours[i])

    # Order the contoursIndexesWithArea array by it's second column
    contoursIndexesWithAreaOrder = contoursIndexesWithArea[contoursIndexesWithArea[:,2].argsort()]

    contoursIndexesWithAreaOrder[:,0] = range(contoursNum)

    return contoursIndexesWithAreaOrder



def getSlopes(array):
    numElements = np.size(array)

    if (numElements < 2):
        return np.zeros(1)

    slopes = np.empty(numElements-1)
    for i in range(numElements-1):
        #print("Element: " + str(i) + " " + str(array[i]) + " " + str(array[i+1]))
        #print("Distance: " + str(array[i] - array[i+1]))
        slopes[i] = np.degrees(np.arctan(array[i]-array[i+1]))
        # print("ArcTan: " + str(array[i]-array[i+1]) + " = " + str(np.degrees(np.arctan(array[i]-array[i+1]))))
    return slopes

def isContourClosed(contour):

    print(np.shape(contour))
    print("Tamaño: " + str(np.size(contour,0)-1))
    #for i in range(np.size(contour,0)):
    for i in range(10):
        print("Point: " + str(i) + " " + str(contour[i]))
        print("Area: " + str(cv2.contourArea(contour)))
        
    exit()


def orderOfMagnitude(number):
    return int(math.log10(number))

def findLargestGap(array):

    biggestGap = 0
    biggestGapIndex = 0
    numElements = np.size(array)
    listOfGaps = []

    if (numElements < 2):
        return 0

    for i in range(numElements-1):
        if ((array[i] - array[i+1]) > biggestGap):
            biggestGapIndex = i
            biggestGap = array[i] - array[i+1]
            listOfGaps.append([biggestGapIndex,biggestGap])

    return biggestGapIndex

def removeImagePerimeterContour(img, contours):

    height,width = img.shape

    contourNum = np.size(contours)
    contourFound = False
    imageArea = height * width

    for i in range(contourNum):
        if ((cv2.contourArea(contours[i]) / imageArea) > 0.80):
            contourFound = True
            break

    if contourFound:
        return np.delete(contours,i)
    else:
        return contours

def clusterContoursKmeans2(contours):

    contoursNum = np.size(contours)

    contoursSizesAndAreas = np.empty(contoursNum*2,dtype='i').reshape(contoursNum,2)
 
    for i in range(contoursNum):
        contoursSizesAndAreas[i,0] = cv2.contourArea(contours[i])
        contoursSizesAndAreas[i,1] = np.size(contours[i])

    ysi = sorted(contoursSizesAndAreas[:,0])
    y = np.float32(ysi)
    
    print("y: "  + str(ysi))

    codebook, _ = kmeans(y, 2)  # three clusters
    cluster_indices, _ = vq(y, codebook)
    print("labels: " + str(cluster_indices))

def clusterContoursKmeans(contours):
    
    contoursNum = np.size(contours)

    contoursSizesAndAreas = np.empty(contoursNum*2,dtype='i').reshape(contoursNum,2)
 
    for i in range(contoursNum):
        contoursSizesAndAreas[i,0] = cv2.contourArea(contours[i])
        contoursSizesAndAreas[i,1] = np.size(contours[i])


    # X = np.random.randint(25,50,(25,2))
    # Y = np.random.randint(60,85,(25,2))

    # X = contoursSizesAndAreas[:,0]
    # Y = contoursSizesAndAreas[:,1]

    X = np.array([[1,2],[2,3],[4,5]])
    Y = np.array([[11,12],[13,14],[15,16]])

    print("X: " + str(X))
    print("Y: " + str(Y))
    print("X Shape: " + str(X.shape))
    print("Y Shape: " + str(Y.shape))
    XY = np.vstack((X,Y))

    print("XY: " + str(XY))
    print("XY Shape: " + str(XY.shape))
    # convert to np.float32
    #Z = np.float32(Z)
    Z = np.float32(sorted(contoursSizesAndAreas[:,0]))
    print("Z: " + str(Z))
    print("Z Shape: " + str(Z.shape))

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    print("Z: " + str(Z))
    print("Z Shape: " + str(Z.shape))

    print("label: " + str(label))
    print("label Shape: " + str(label.shape))

    return label

    # # Now separate the data, Note the flatten()
    # A = Z[label.ravel()==0]
    # B = Z[label.ravel()==1]
    # # Plot the data
    # plt.scatter(A[:,0],A[:,1])
    # plt.scatter(B[:,0],B[:,1],c = 'r')
    # plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    # plt.xlabel('Height'),plt.ylabel('Weight')
    # plt.show()

def clusterContoursKmeans3(contours):
    
    contoursNum = np.size(contours)

    contoursSizesAndAreas = np.empty(contoursNum*2,dtype='i').reshape(contoursNum,2)
 
    for i in range(contoursNum):
        contoursSizesAndAreas[i,0] = cv2.contourArea(contours[i])
        contoursSizesAndAreas[i,1] = np.size(contours[i])


    # X = np.random.randint(25,50,(25,2))
    # Y = np.random.randint(60,85,(25,2))

    X = sorted(contoursSizesAndAreas[:,0])
    Y = sorted(contoursSizesAndAreas[:,1])


    print("X: " + str(X))
    print("Y: " + str(Y))
    XY = np.vstack((X,Y))

    print("XY: " + str(XY))
    print("XY Shape: " + str(XY.shape))
    # convert to np.float32
    #Z = np.float32(Z)
    Z = np.float32(XY)
    print("Z: " + str(Z))
    print("Z Shape: " + str(Z.shape))

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    print("Z: " + str(Z))
    print("Z Shape: " + str(Z.shape))

    print("label: " + str(label))
    print("label Shape: " + str(label.shape))

    # Now separate the data, Note the flatten()
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]
    # Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()
