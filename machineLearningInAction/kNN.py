from numpy import *
import operator

def createDataSet():
    group = array([ [1.0, 1.1], [1.0,1.0], [0, 0], [0, 0.1] ])
    labels = ['A', 'B', 'B', 'B']
    return group, labels

def calcDistance(inX, dataSet, labels, k):
    # shape is a tuple that gives dimensions of the array
    # shape[0] returns the number of rows, here will return 4
    dataSetSize = dataSet.shape[0]  # dataSetSize = 4

    # numpy.tile(A, reps): construct an array by repeating A the number of times given by reps
    # if reps has length d, the result will have dimension of max(d, A.ndim)
    # tile(inX, (dataSetSize,1)) will return [ [0,0] [0,0] [0,0] [0,0] ]
    # diffMat is [ [-1, -1.1], [-1, -1], [0, 0], [0, -0.1] ]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # **2 means square
    sqDiffMat = diffMat ** 2

    # sqDistances = x^2 + y^2
    sqDistances = sqDiffMat.sum(axis=1)
    # distance is equal to the square root of the sum of the squares of the coordinates
    # distance = [1.48, 1.41, 0, 0.1]
    distances = sqDistances ** 0.5

    # numpy.argsort() returns the indices that would sort an array
    # here returns [2 3 1 0]
    sortedDistIndices = distances.argsort()
    return sortedDistIndices

def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
    classCount = {}

    # iterate k times from the closest item
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        # increase +1 on the selected label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort ClassCount Descending order
    return sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)


def classify0(inX, dataSet, labels, k):
    # calculate the distance between inX and the current point
    sortedDistIndices = calcDistance(inX, dataSet, labels, k)
    # take k items with lowest distances to inX and find the majority class among k items
    sortedClassCount = findMajorityClass(inX, dataSet, labels, k, sortedDistIndices)
    # sortedClassCount is now [('B', 3)], therefore return B
    return sortedClassCount[0][0]


group, labels = createDataSet()
result = classify0([0,0], group, labels,3)
print result