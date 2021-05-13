# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:54:37 2015

@author: erhe01
"""

import numpy as np
from . import GenericAlgorithms
import scipy.interpolate as si
import math


class CatmullRomSpline():
    '''
    spline that goes through control points with arc length mapping used by motion planning
    implemented using the following resources and examples:
    #http://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    #http://algorithmist.net/docs/catmullrom.pdf
    #http://www.mvps.org/directx/articles/catmull/
    #http://hawkesy.blogspot.de/2010/05/catmull-rom-spline-curve-implementation.html
    #http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
    '''
    def __init__(self,controlPoints, dimensions, granularity=100):
        self.granularity = granularity
        #http://algorithmist.net/docs/catmullrom.pdf
        #base matrix to calculate one component of a point on the spline based on the influence of control points
        self.catmullRomBaseMatrix = np.array([[-1.0, 3.0, -3.0, 1.0],
                                              [2.0, -5.0, 4.0, -1.0],
                                              [-1.0, 0.0, 1.0, 0.0],
                                              [0.0, 2.0, 0.0, 0.0]])
        self.dimensions = dimensions
        self.fullArcLength = 0
        self.initiated = False
        self.controlPoints = []
        self.numberOfSegments = 0
        if len (controlPoints) >0:
            self.initiateControlPoints(controlPoints)
            self.initiated = True


    def initiateControlPoints(self,controlPoints):
        '''
        @param controlPoints array of class accessible by controlPoints[index][dimension]
        '''
        self.numberOfSegments = len(controlPoints)-1
        self.controlPoints = [controlPoints[0]]+controlPoints+[controlPoints[-1],controlPoints[-1]]#as a workaround add multiple points at the end instead of one
        print("length of control point list ",len(self.controlPoints))
        print("number of segments ",self.numberOfSegments)
        print("number of dimensions",self.dimensions)


        self.updateArcLengthMappingTable()

        return

    def addPoint(self,point):

        #add point replace auxiliary control points
        if  self.initiated:
            del self.controlPoints[-2:]
            self.numberOfSegments = len(self.controlPoints)-1#"-2 + 1
            self.controlPoints += [point,point,point]
            # print(self.controlPoints)

            #update arc length mapping
            self.updateArcLengthMappingTable()
        else:
            #print "here",point
            self.initiateControlPoints([point,])
            self.initiated = True


    def clear(self):
        self.controlPoints = []#[startPoint,startPoint,startPoint]#as a workaround add multiple points at the end instead of one]
        self.initiated = False
        self.fullArcLength = 0
        self.numberOfSegments = 0
        self.arcLengthMap = []

    def transformByMatrix(self,matrix):
        '''
        matrix nxn transformation matrix where n is the number of dimensions of the catmull rom spline
        '''
        if self.dimensions < matrix.shape[0]:
            for i in range(len(self.controlPoints)):
                self.controlPoints[i] = np.dot(matrix, self.controlPoints[i])#+[,1][:4]
    #             print "t",i
        else:
            print("failed",matrix.shape)
        return

    def updateArcLengthMappingTable(self):
        '''
        creates a table that maps from parameter space of query point to relative arc length based on the given granularity in the constructor of the catmull rom spline
        http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
        '''
        self.fullArcLength = 0
        granularity = self.granularity
        u = np.arange(granularity+1) / float(granularity)
        lastPoint = None
        numberOfEvalulations = 0
        self.arcLengthMap = []
        for i in u:
            point = self.queryPoint(i)
            if lastPoint is not None:
                delta = []
                d = 0
                while d < self.dimensions:
                    delta.append(math.sqrt((point[d]-lastPoint[d])**2))
                    d += 1
                self.fullArcLength += np.sum(delta)#(point-lastPoint).length()
                #print self.fullArcLength
            self.arcLengthMap.append([i,self.fullArcLength])
            numberOfEvalulations+=1
            lastPoint= point

       # self.fullArcLength = arcLength
        print("len",self.fullArcLength)
        print("points",numberOfEvalulations)
        #normalize values
        if self.fullArcLength > 0 :
            for i in range(numberOfEvalulations):
                self.arcLengthMap[i][1] /= self.fullArcLength

        # print(self.arcLengthMap)
        return

    def getFullArcLength(self, granularity = 100):
        #granularity = self.granularity
        u = np.arange(granularity+1) / float(granularity)
        arcLength = 0.0
        lastPoint = None
        for i in  u:
            # print("sample",i)
            point = self.queryPoint(i)
            if lastPoint != None:
                arcLength += np.linalg.norm(point-lastPoint)#(point-lastPoint).length()
                lastPoint= point
                # print(point)
#             else:
#                 print "point is None"
        return arcLength

    def getDistanceToPath(self,absoluteArcLength, position):
        '''
        evaluates a point with absoluteArcLength on self to get a point on the path
        then the distance between the given position and the point on the path is returned
        '''
        pointOnPath = self.getPointAtAbsoluteArcLength(absoluteArcLength)
        return np.linalg.norm(position-pointOnPath)

    def getLastControlPoint(self):
        if len(self.controlPoints)> 0:
            return self.controlPoints[-1]
        else:
            return [0,0,0]

    def getArcLengthForParameter(self,t):
        stepSize = 1/self.granularity
        tableIndex = int(t/stepSize)
        return self.arcLengthMap[tableIndex][1]*self.fullArcLength


    def getPointAtAbsoluteArcLength(self,absoluteArcLength):
        point = np.zeros((1,self.dimensions))#source of bug
        if absoluteArcLength <= self.fullArcLength:
            # parameterize curve by arc length
            relativeArcLength = absoluteArcLength/self.fullArcLength
            point = self.queryPointByRelativeArcLength(relativeArcLength)
        else:
            return None
#         else:
#             raise ValueError('%f exceeded arc length %f' % (absoluteArcLength,self.fullArcLength))
        return point

    def findClosestValuesInArcLengthMap(self,relativeArcLength):
        '''
        - given a relative arc length between 0 and 1 it uses closestLowerValueBinarySearch from the Generic Algorithms module to search the self.arcLengthMap for the values bounding the searched value
        - returns floor parameter, ceiling parameter, floor arc length, ceiling arc length and a bool if the exact value was found
        '''
        foundExactValue = True
        result = GenericAlgorithms.closestLowerValueBinarySearch(self.arcLengthMap,0,len(self.arcLengthMap)-1,relativeArcLength, getter = lambda A,i: A[i][1])#returns the index and a flag value, requires a getter for the array
        #print result
        index = result[0]

        if result[1] == 0:#found exact value
            floorP, ceilP = self.arcLengthMap[index][0],self.arcLengthMap[index][0]
            floorL, ceilL = self.arcLengthMap[index][1],self.arcLengthMap[index][1]
            foundExactValue = True
        elif result[1] ==1:#found lower value
            floorP = self.arcLengthMap[index][0]
            floorL = self.arcLengthMap[index][1]
            if index <len(self.arcLengthMap):#check array bounds
                ceilP = self.arcLengthMap[index+1][0]
                ceilL = self.arcLengthMap[index+1][1]
                foundExactValue = False
            else:
                foundExactValue = True
                ceilP= floorP
                ceilL = floorL
        elif result[1] ==2:#value smaller than smallest element in the array
            ceilP = self.arcLengthMap[index][0]
            floorL = self.arcLengthMap[index][1]
            floorP  = ceilP
            ceilL = floorL
            foundExactValue = True
        elif result[1] ==3:#value larger than largest element in the array
            ceilP = self.arcLengthMap[index][0]
            ceilL = self.arcLengthMap[index][1]
            floorP = ceilP
            floorL = ceilL
            foundExactValue = True
        #print relativeArcLength,floorL,ceilL,foundExactValue
        return floorP,ceilP,floorL,ceilL,foundExactValue

    #see slide 30 of http://pages.cpsc.ucalgary.ca/~jungle/587/pdf/5-interpolation.pdf
    #note it does a binary search so it is rather expensive to be called at every frame
    def queryPointByRelativeArcLength(self,relativeArcLength):

        floorP,ceilP,floorL,ceilL,foundExactValue = self.findClosestValuesInArcLengthMap(relativeArcLength)
        if not foundExactValue:
            alpha = (relativeArcLength-floorL)/(ceilL-floorL)#can be reused a-
            #t = floorL+alpha*(ceilL-floorL)
            t = floorP+alpha*(ceilP-floorP)
        else:
            t = floorP
        #t = relativeArcLength#todo add correct mapping

        return self.queryPoint(t)

    def mapToSegment(self,t):

        i =  min(math.floor( self.numberOfSegments *t),self.numberOfSegments)#the part of t before i
        localT =(self.numberOfSegments*t) -math.floor( self.numberOfSegments *t)#the rest, e.g. N = 10 and t = 0.62 => i = 6 and the rest is 0.02
        #i = min(i,self.numberOfSegments)
        return i+1,localT#increment i by 1 to ignore the first auxiliary control point


    def getControlPointVectors(self,i):
        i = int(i)
        #if i<=self.numberOfSegments-2:
        d = 0
        vectors = []

        while d < self.dimensions:
            v = [float(self.controlPoints[i-1][d]),float(self.controlPoints[i][d]),float(self.controlPoints[i+1][d]),float(self.controlPoints[i+2][d])]
            vectors.append(np.array(v))
            d+=1

        return vectors

#
    def queryPoint(self, t):
        i,localT = self.mapToSegment(t)
        weightVector = np.array([localT**3,localT**2,localT,1])
        controlPointVectors = self.getControlPointVectors(i)
        point =[]
        d =0
        while d < self.dimensions:
            point.append(self.queryValue(weightVector, controlPointVectors[d]))
            d += 1
        return np.array(point)

    def queryValue(self, weightVector, controllPointVector):
        v = np.dot(self.catmullRomBaseMatrix, controllPointVector)
        v = np.dot(weightVector, v)
        return 0.5 * v
