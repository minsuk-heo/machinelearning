from numpy import *
import operator

def createDataSet():
    group = array([ [1.0, 1.1], [1.0,1.0], [0, 0], [0, 0.1] ])
    labels = ['A', 'B', 'B', 'B']
    return group, labels

