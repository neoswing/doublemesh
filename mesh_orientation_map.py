#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:19:07 2023

@author: melnikov
"""

import numpy
import json
from scipy import ndimage, spatial
import scipy.cluster.hierarchy as hca
import base64
from matplotlib import pyplot as plt
import multiprocessing as mp
import ctypes


try:
    from workflow_lib import workflow_logging
    logger = workflow_logging.getLogger()
except:
    import logging
    logger = logging.getLogger("MeshBest")



def From64ToSpotArray(string64):
    array = numpy.frombuffer(base64.b64decode(string64))
    array = array.reshape((int(array.size/5), 5))

    return array


def DistanceCalc(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):

            
    count1 = spots1.shape[0]
    count2 = spots2.shape[0]

    RealCoords1 = numpy.zeros((count1, 3))
    RealCoords2 = numpy.zeros((count2, 3))
    

    x = (spots1[:, 1] - BeamCenter[0]) * DetectorPixel
    y = (spots1[:, 2] - BeamCenter[1]) * DetectorPixel
    divider = Wavelength * numpy.sqrt(x ** 2 + y ** 2 + DetectorDistance ** 2)
    RealCoords1[:, 0] = x / divider
    RealCoords1[:, 1] = y / divider
    RealCoords1[:, 2] = (1/Wavelength) - DetectorDistance/divider
    
    x = (spots2[:, 1] - BeamCenter[0]) * DetectorPixel
    y = (spots2[:, 2] - BeamCenter[1]) * DetectorPixel
    divider = Wavelength * numpy.sqrt(x ** 2 + y ** 2 + DetectorDistance ** 2)
    RealCoords2[:, 0] = x / divider
    RealCoords2[:, 1] = y / divider
    RealCoords2[:, 2] = (1/Wavelength) - DetectorDistance/divider


        

    output = []
    
    thrsh = 0.1*3.14159/180.0

    
    DistanceMatrix = spatial.distance.cdist(RealCoords1, RealCoords2, metric='euclidean')
#    print(DistanceMatrix)
    if count1 >= count2:
        output = numpy.min(DistanceMatrix, axis=0)
    else:
        output = numpy.min(DistanceMatrix, axis=1)
    
    
    output[output>thrsh] = thrsh

    output = numpy.array(output)

    F = numpy.sqrt(numpy.mean(output**2))*180/3.14159
    return F


def slicer3x3(key):
    mask = numpy.meshgrid(numpy.arange(-1, 2), numpy.arange(-1, 2))
    mask[0] += key[0]
    mask[1] += key[1]
    return mask
    

def CalculateZoneH(jsondata, keys, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):

    keys = numpy.array(keys).T

    for item in keys:
        slicer = slicer3x3(item)
        print(slicer)


#    T = T+numpy.transpose(T)
#    for key, value in numpy.ndenumerate(T):
#        if value!=value:
#            ar = numpy.maximum(T[key[0], :], T[:, key[1]])
#            if numpy.all(numpy.isnan(ar)):
#                T[key] = 0.1
#            else:
#                T[key] = numpy.nanmean(numpy.maximum(T[key[0], :], T[:, key[1]]))
#
##    logger.debug('constructed distance matrix')
#
#
#
#    plt.imshow(T, interpolation='nearest', cmap='hot')
#    plt.colorbar()
##    plt.savefig('Dmatrix.png', dpi=300, bbox_inches='tight')
#    plt.close()
#    
##    numpy.savetxt('Dmatrix.txt', T, fmt='%.3f')
#
#    q = []
#    for key in numpy.ndenumerate(T):
#        if key[0][1] > key[0][0]:
#            q.append(key[1])
#    T = numpy.array(q)
#
#
#



def PerformCrystalRecognition(jsondata):

    
    Wavelength = jsondata['wavelength']
    DetectorDistance = jsondata['detector_distance']
    BeamCenter = (jsondata['orgx'], jsondata['orgy'])
    DetectorPixel = 0.075
    
    print(Wavelength, DetectorDistance, BeamCenter)


    row = jsondata['steps_y']
    col = jsondata['steps_x']
    positionReference = numpy.empty((row, col), dtype='int')
    for i in jsondata['meshPositions']:
        positionReference[i['indexZ'], i['indexY']] = i['index']
    

    Dtable = numpy.frombuffer(base64.b64decode(jsondata['MeshBest']['Dtable'])).reshape((row, col))
    
    
    
    plt.imshow(Dtable, cmap='hot')
    plt.show()
    
    Htable = -numpy.zeros(Dtable.shape)


    ZoneMap = ndimage.measurements.label((Dtable > 0), structure=numpy.ones((3, 3)))

    for z in range(1, 1+ZoneMap[1]):
        keys = numpy.where(ZoneMap[0]==z)
        if len(keys[0])==1:
            Htable[keys[0][0], keys[1][0]] = 0
        else:
            keys = numpy.array(keys).T

            for item in keys:
                hlist = []
                slicer = slicer3x3(item)
                for key in numpy.ndindex((3, 3)):
                    if not (key[0]==1 and key[1]==1):
                        if (slicer[0][key]>=0 and slicer[0][key]<row) and (slicer[1][key]>=0 and slicer[1][key]<col) and Dtable[slicer[0][key], slicer[1][key]]>0.3:
#                            print(item)
#                            print(slicer[0][key], slicer[1][key])
                            key1 = positionReference[item[0], item[1]]
                            key2 = positionReference[slicer[0][key], slicer[1][key]]
                            
                            spots1 = From64ToSpotArray(jsondata['meshPositions'][key1]['dozorSpotList'])
                            spots2 = From64ToSpotArray(jsondata['meshPositions'][key2]['dozorSpotList'])

                            hlist.append(DistanceCalc(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel))
#                            print(DistanceCalc(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel))
                
                Htable[item[0], item[1]] = numpy.mean(hlist)

    plt.imshow(Htable, cmap='hot')
    plt.colorbar()
    plt.show()





#direc = '/home/esrf/melnikov/spyder/test/Workflow_20230121-153857/test_meshbest/'
#direc = '/home/esrf/melnikov/spyder/test/Workflow_20210901-101908/test_meshbest/'
#direc = '/home/esrf/melnikov/spyder/test/Workflow_20230125-091931/test_meshbest/'
direc = '/home/esrf/melnikov/spyder/test/Workflow_20230227-171902/test_meshbest/'


with open(direc+'MeshResults.json', 'r') as f:
    jsondata = json.load(f)



PerformCrystalRecognition(jsondata)





