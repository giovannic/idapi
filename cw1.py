#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import zeros, array, divide, dot
from numpy.linalg import eig 
from collections import Counter
from operator import mul

#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here

    mean = sum(realData) / len(theData)

    # Coursework 4 task 1 ends here
    return array(mean)

def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    u = realData - Mean(theData)
    covar = u.transpose().dot(u) / (len(theData) - 1)

    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis):
    # Coursework 4 task 3 begins here
    count = 0
    for f in theBasis:
        SaveEigenface(f, "PrincipalComponent%d.jpg" % count)
        count = count + 1

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    return (theFaceImage - theMean).dot(theBasis.transpose())

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    # Coursework 4 task 5 begins here
    x = array(aMean)
    count = 0
    SaveEigenface(x, "Mean.jpg")
    for (phi, mag) in zip(aBasis, componentMags):
        x = dot(phi, mag) + x
        count = count + 1
        SaveEigenface(x, "MeanPlus%d.jpg" % count)

def MyCreate(aBasis, aMean, componentMags):
    # Coursework 4 task 5 begins here
    x = array(aMean)
    count = 0
    SaveEigenface(x, "Mean.jpg")
    for (phi, mag) in zip(aBasis, componentMags):
        x = dot(phi, mag) + x
        count = count + 1
        SaveEigenface(x, "MyPlus%d.jpg" % count)
    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    realData = theData.astype(float)
    u = realData - Mean(realData)
    smallm = u.dot(u.transpose())
    w, v = eig(smallm)
    vs = zip(w,u.transpose().dot(v).transpose())
    vs.sort(key=lambda t:t[0], reverse=True)
    orthoPhi = map(lambda t: t[1], vs)
    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 4
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)

images = array(ReadImages())
meanImage = ReadOneImage("MeanImage.jpg")
c = array(ReadOneImage("c.pgm"))
theBasis = ReadEigenfaceBasis()
CreateEigenfaceFiles(theBasis)
cProjection = ProjectFace(theBasis, meanImage, c)
CreatePartialReconstructions(theBasis, meanImage, cProjection)

myBasis = PrincipalComponents(images)
myprojection = ProjectFace(myBasis, meanImage, c)
MyCreate(myBasis, meanImage, myprojection)
