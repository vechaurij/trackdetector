import numpy as np

def getContoursOrdered(contours):

    contoursNum = np.size(contours)

    contoursIndexesWithSize = np.empty(contoursNum*2,dtype='i').reshape(contoursNum,2)
 
    for i in range(contoursNum):
        contoursIndexesWithSize[i,0] = i
        contoursIndexesWithSize[i,1] = np.size(contours[i])

    # Order the contoursIndexesWithSize array by it's second column
    contoursIndexesWithSizeReverseOrder = contoursIndexesWithSize[contoursIndexesWithSize[:,1].argsort()][::-1]

    return contoursIndexesWithSizeReverseOrder[:,0]
    
