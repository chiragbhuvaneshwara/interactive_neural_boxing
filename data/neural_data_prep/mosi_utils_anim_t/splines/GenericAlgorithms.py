# -*- coding: utf-8 -*-
#===============================================================================
# author: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realität)
# some simple algorithms partly copied directly from pseudocode
# last update: 19.3.2014
#===============================================================================

 
import numpy as np
import scipy.signal as signal
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt
    
#http://www.peterbe.com/plog/uniqifiers-benchmark
def uniquifyList(seq,key = lambda x: x):
   # order preserving
#    if idfun is None:
#        def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = key(item)#item[1]#idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result 


#testTree()
def merge(A,B):
    C =[]
    while len(A) >0 or len(B)>0:
        if len(A) >0 and len(B)>0:    
            if A[0]<=B[0]:
                C.append(A[0])
                A.pop(0)
            else:
                C.append(B[0])
                B.pop(0)
  
        elif len(A) >0:
             C.append(A[0])
             A.pop(0)
        elif len(B) >0:
             C.append(B[0])
             B.pop(0)

        
    return C


'http://en.wikipedia.org/wiki/Mergesort'
#top down implementation
#O(log n)
def mergesort(L):
    #print integerArray
    arraySize= len(L)
    if arraySize<=1:
        return L
    else:
        halfSize = int(arraySize/2)
        a = mergesort(L[:halfSize])
        b = mergesort(L[halfSize:])
    return  merge(a,b)
 

'http://en.wikipedia.org/wiki/Quicksort'
#O(log n)
def naiveQuicksort(A):
    
    if len(A)<=1:
        return A
    pivotIndex = int(len(A)/2)
    pivot= A.pop(pivotIndex)
    less=[]
    greaterEqual=[]
    for x in A:
        if x <pivot:
            less.append(x)
        else:
            greaterEqual.append(x)
    return naiveQuicksort(less)+[pivot]+naiveQuicksort(greaterEqual)

#returns new pivot index that divides the input array in parts smaller and greater than the pivot value
'http://en.wikipedia.org/wiki/Quicksort'
def partition(A,left,right,pivot):
    #print A
    pivotValue = A[pivot]
    #move pivot to the end so it does not get in the way
    A[pivot] =  A[right] 
    A[right] = pivotValue
     
    storeIndex = left
    i = left
    #print A
    while i < right:
        if A[i]<=pivotValue:
             temp =  A[i]
             A[i] =  A[storeIndex] 
             A[storeIndex] = temp
             storeIndex +=1#increase pivot index/center as long as points are smaller 
             #print A
             #print i
        i +=1
        
    #swap pivot element with store index
    temp = A[storeIndex]
    A[storeIndex] =  A[right] 
    A[right] = temp
    #print A
    return storeIndex
       
    
#http://www.youtube.com/watch?v=V6mKVRU1evU
def partition2(A,left,right,pivot): 
    i = left
    j = right
    #pivotValue = A[pivot]
    while True:
        #print "smaller"
        while A[i]<A[pivot]:# and i < right:
            #print A[i]
            i +=1
        #print "greater"
        while A[j]>A[pivot] and j > 0:#=left:
            #print A[j]
            j-=1
        if i >=j:#termination condition
            break
        else:
            #swap value on left side of pivot with value on right side of pivot
            temp = A[i]
            A[i] = A[j]
            A[j] = temp
            
    #print i
#     temp = A[i]
#     A[i] = A[right]
#     A[right] = temp
    return i

    
'http://en.wikipedia.org/wiki/Quicksort'
#O(n log n)
#superficially
# number of comparisons = log n! 
# number of comparisons = log n + log(n-1) + ... + log(1)
# number comparisons = n log n 
def inPlaceQuicksort(A,left,right):
    if left < right:
        pivot =right#left#(right-left)/2
        newPivot = partition2(A,left,right,pivot)
        #print A
        #print newPivot

        inPlaceQuicksort(A,left,newPivot-1)#recursively sort left part
        inPlaceQuicksort(A,newPivot+1,right)#recursively sort right part
        
    return A
    
#O(log n)
def iterativeBinarySearch(A,value):
    #print A
    #print value
    #value= A[1]
    min = 0
    max = len(A)-1
    while min <= max:
        mid = min+((max-min)/2)
        print(mid)
        if A[mid]<value:
            min = mid+1
        elif A[mid] > value:
            max = mid-1
        else:
            return mid
    
    return
    
#http://stackoverflow.com/questions/2307283/what-does-olog-n-mean-exactly
#http://en.wikipedia.org/wiki/Binary_search_algorithm
#O(log n)
def binarySearch(A,left,right,value,getter= lambda A,i : A[i]):
    #print value
    #print left,right
    if  right < left:
        return -1
    
    else:
        if left ==right:
           if  A[left]==value:
               return left
           
           else:
               return -1
        
        else:
            try:
                iMid = int(left+((right-left)/2))
                print(iMid)
                if A[iMid]>value:
                    #value is in left part
                    print("left",value,A[iMid])
                    return binarySearch(A,left,iMid-1,value)
                elif A[iMid]<value:
                    #value is in right part
                    print("right",value,A[iMid])
                    return  binarySearch(A,iMid+1,right,value)
                else:
                    #found value
                    return iMid
            except:
                print("exception ",left,right,value,A)
   

#COPIED from http://stackoverflow.com/questions/4257838/how-to-find-closest-value-in-sorted-array
def closestLowerValueBinarySearch(A,left,right,value,getter= lambda A,i : A[i]):   
    '''
    - left smallest index of the searched range
    - right largest index of the searched range
    - A array to be searched
    - parameter is an optional lambda function for accessing the array
    - returns a tuple (index of lower bound in the array, flag: 0 = exact value was found, 1 = lower bound was returned, 2 = value is lower than the minimum in the array and the minimum index was returned, 3= value exceeds the array and the maximum index was returned)
    '''

    #result =(-1,False)
    delta = int(right -left)
    #print delta
    if (delta> 1) :#or (left ==0 and (delta> 0) ):# or (right == len(A)-1 and ()):#test_interactive if there are more than two elements to explore
        iMid = int(left+((right-left)/2))
        testValue = getter(A,iMid)
        #print "getter",testValue
        if testValue>value:
            #print "right"
            return closestLowerValueBinarySearch(A, left, iMid, value,getter)
        elif testValue<value:
            #print "left"
            return closestLowerValueBinarySearch(A, iMid, right, value,getter)
        else:
            #print "done"
            return (iMid,0)
    else:#always return the lowest closest value if no value was found, see flags for the cases
        leftValue = getter(A,left)
        rightValue = getter(A,right)
        if value >= leftValue:
            if value <= rightValue:
                return (left,1)
            else:
                return (right,2)
        else:
            return(left,3)
    
    #return result

def notBubblesort(A):
    i = 0
    while i < len(A):
        j = 0
        while j < len(A):
            if  A[i] <A[j]:
                temp = A[j]
                A[j] = A[i]
                A[i] = temp
            j+=1
        i+=1
        
    return A
                
    
#O(n^2)
def bubblesort(A):
    border = len(A)
    while border> 0:
        i = 0
        while i < border-1:#compare A[i] with every ai part of A | i < border
            neighbor = i+1
            #test_interactive if A[i] smaller than A[i+1]
            if  A[i] >A[neighbor]:
                #swap with neighbor
                temp = A[neighbor]
                A[neighbor] = A[i]
                A[i] = temp
            i+=1
        border-=1#move border to the left
        
    return A     
        
#         
# [1,3,2]
# A[2]=2
# i = 1
# A[2]!>2

def insertionsort(A):
    j = 1
    while j < len(A):
        # sort A[0,..,j] while assuming that A[0,...,j-1] is sorted
        value = A[j] #store current value
        # compare with all preceding values until a value is found that is larger than the current value
        # then swap it with the current value
        i = j-1
        while i >= 0 and A[i] > value:
            #swap with preceeding value
            A[i+1]=A[i] #self note A[i=j] is stored in value and can be overwritten
            #and reduce index
            i -= 1
        # put the stored value to the position of the last swap
        A[i+1]=value 
        j+=1
        
    return A
    



#COPIED from http://wiki.scipy.org/Cookbook/SignalSmooth
def discreteGaussKernel(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """

    size = int(size)
    if not sizey:
         sizey = size
    else:
         sizey = int(sizey)
    #print size, sizey
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()
 
#COPIED from http://wiki.scipy.org/Cookbook/SignalSmooth
def blur2DImage(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
       size in the y direction.
    """
    g = discreteGaussKernel(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)
 
 
#COPIED from http://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficie
#The location of the local minima can be found for an array of arbitrary dimension using Ivan's detect_peaks function, with minor modifications:
def detectLocalMinima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local minimum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood minimum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    #print eroded_background
    #print detected_minima.shape
    return local_min,np.where(detected_minima)   

#note due to some misunderstanding on my part of the matplotlib output the x and y are inserted in the opposite order
def extractFilteredMinima(distanceMatrix, thresholdFactor):
    local_min,localMinCoords = detectLocalMinima(distanceMatrix)  
    
    #find global minimum
    globalMin = 10000.0
    index= 0
    while index < len(localMinCoords[0]):
        x =localMinCoords[1][index]
        y =localMinCoords[0][index]
        min = distanceMatrix[y][x]
        #min = distanceMatrix[x][y]
        #print min, min2
        if globalMin > min:
            globalMin = min
            
        index+=1
    #filter local minima using the threshold
    optimalMinCoords =[]
    optimalMinCoords.append([])
    optimalMinCoords.append([])
    index = 0
    while index < len(localMinCoords[0]):
        x =localMinCoords[1][index]
        y =localMinCoords[0][index]
        #min = distanceMatrix[x][y]
        min = distanceMatrix[y][x]
        if min < globalMin+(globalMin*thresholdFactor):
            optimalMinCoords[1].append(x)
            optimalMinCoords[0].append(y)
        index +=1

#     globalMin = 10000.0
#     for x in localMinCoords[1]:
#         for y in localMinCoords[0]:
#             min = distanceMatrix[y][x]
#             #min = distanceMatrix[x][y]
#             #print min, min2
#             if globalMin > min:
#                 globalMin = min
#     #filter local minima using the threshold
#     optimalMinCoords =[]
#     optimalMinCoords.append([])
#     optimalMinCoords.append([])
#     for x in localMinCoords[1]:
#         for y in localMinCoords[0]:
#             min = distanceMatrix[y][x]
#             #min = distanceMatrix[x][y]
#             if min < globalMin+(globalMin*thresholdFactor):
#                 optimalMinCoords[1].append(x)
#                 optimalMinCoords[0].append(y)
                
    
    return optimalMinCoords
    
#http://docs.scipy.org/doc/scipy/reference/cluster.html
#http://scikit-learn.org/stable/modules/clustering.html
#http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#example-cluster-plot-mean-shift-py
def extract2DClusters(data):
    bandwidth = estimate_bandwidth(data, quantile=0.3, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(cluster_centers)
    print(labels)
#     labels_unique = np.unique(labels)
#     n_clusters_ = len(labels_unique)
    print(("number of estimated clusters : %d" % len(cluster_centers)))
    #KMeans(init='k-means++', n_clusters=numOfClusters, n_init=10)
    return cluster_centers,labels

def filterClusters(data,costList):
    #get cluster centers using scipy mean shift implementation
    cluster_centers, labels=extract2DClusters(data)
    #create a map that contains for each cluster a list of the associated points with indices in the input data and the associated cost
    clusterToPointsMap = {}
    maxKey = len(cluster_centers)
    print("cluster keys")
    ci = 0
    while ci <=maxKey:
        print(ci)
        clusterToPointsMap[ci] = []
        ci+=1
    
    print("append point indices")
   #insert indices into cluster map
    for li in range(0,len(labels)-1):
        #clusterTuple = li,cost[li]
        if labels[li] >= 0 and labels[li] <=maxKey:
            #print labels[li]
            clusterToPointsMap[labels[li]].append(li)
      
      
    filteredData =[],[]
    #filter the points with the least cost for each cluster
    for key in list(clusterToPointsMap.keys()):
        #find minumum and append to the output list
        if len(clusterToPointsMap[key])>0:
            minIndex =clusterToPointsMap[key][0] 
            minCost = costList[minIndex]
            for i in range(1,len(clusterToPointsMap[key])):
                cost = costList[clusterToPointsMap[key][i]]   
                if cost<minCost:
                    minIndex = clusterToPointsMap[key][i]
                    minCost = cost 
            #copy from the original input data into the output data
            filteredData[0].append(data[minIndex][0])
            filteredData[1].append(data[minIndex][1])
        else:
            print("no elements in list for key ",key)
    return filteredData

#todo change: filter minima by selecting min in a window
        
            

def createDistanceMatrixWithMinimaFigure(distanceMatrix,localMinYXCoords,xLabel,yLabel,figureNumber):
    plt.figure(figureNumber)
    plt.imshow(distanceMatrix)#, interpolation='nearest'
    plt.ylabel(xLabel ,fontsize= 16)
    plt.xlabel(yLabel ,fontsize= 16)
    cbar = plt.colorbar(ticks = [np.min(distanceMatrix),np.max(distanceMatrix)])
    cbar.solids.set_edgecolor("face")
    plt.scatter(localMinYXCoords[1],localMinYXCoords[0],c="r")
    np.save("distance_matrix", distanceMatrix)
#     #print distanceMatrix
#     plt.figure(1)
#     plt.imshow(blurredDistanceMatrix)#, interpolation='nearest'
#     plt.ylabel(ntpath.split(file1)[-1],fontsize= 16)
#     plt.xlabel(ntpath.split(file2)[-1],fontsize= 16)
#     cbar = plt.colorbar(ticks = [np.min(blurredDistanceMatrix),np.max(blurredDistanceMatrix)])
#     cbar.solids.set_edgecolor("face")
#     plt.scatter(blocalMinCoords[1],blocalMinCoords[0],c="r")
#     print localMin