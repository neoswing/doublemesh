#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:58:53 2023

@author: melnikov
"""

import numpy
import os
import json
import time
import base64
from scipy import stats, spatial
import matplotlib.pyplot as plt



import logging
logger = logging.getLogger("MeshBest")
logger.setLevel(logging.DEBUG)
for h in logger.handlers:
    h.close()
    logger.removeHandler(h)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
logger.propagate = False



def Spherical_coords(xyz):
    ptsnew = numpy.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = numpy.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = numpy.arctan2(numpy.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = numpy.arctan2(xyz[:,2], numpy.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = numpy.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def triangle(x0, y0, length):
    x = numpy.linspace(0, 99, 100)
    array = y0 - 2 * y0 * numpy.abs(x - x0) / length
    array = array * (array > 0)
    return array


def From64ToSpotArray(string64):
    array = numpy.frombuffer(base64.b64decode(string64))
    array = array.reshape((int(array.size/5), 5))
    return array


def AMPDiter(array):


    L = int(len(array) / 2)
    matrix = numpy.zeros((L, len(array)))

    for k in range(1, L + 1):
        for i in range(1, len(array) + 1):

            if i >= k + 2 and i < len(array) - k + 2:
#                W = 2 * k
                if array[i - 2] > array[i - k - 2] and array[i - 2] > array[i + k - 2]:
                    matrix[k - 1, i - 1] = 0
                else:
                    matrix[k - 1, i - 1] = 1 + numpy.random.random() / 2
            else:
                matrix[k - 1, i - 1] = 1 + numpy.random.random() / 2
    gammas = numpy.sum(matrix, axis=1)
#    logger.debug(gammas)



    Lambda = numpy.where(gammas == numpy.min(gammas))[0][0] + 1
#    logger.debug(Lambda)

    matrix = matrix[:Lambda, :]
    Sigma = numpy.std(matrix, axis=0)

    peaks = []
    for i in range(len(Sigma)):
        if Sigma[i] == 0:
            if (i - 1) / float(len(array)) > 0.00:
                peaks.append(i - 1)
    peaks = numpy.array(peaks, dtype=int)
#    logger.debug('AMPD-result_PEAKS: ', peaks)
    return peaks, Lambda

def AMPD(array_orig):

    fullpeaklist = numpy.array([], dtype=int)

    for cycle in range(10):
        M = numpy.mean(array_orig)
#        SD = numpy.std(array_orig)
        X = numpy.arange(0, len(array_orig))
        linfit = stats.linregress(X, array_orig)
        array = array_orig - (linfit[0] * X + linfit[1])
        array = array * (array > 0)
        MAX = numpy.max(array) - M

        allpeaks = numpy.array([], dtype=int)

        while True:
            substract = numpy.zeros(len(array_orig))
            peaks, Lambda = AMPDiter(array)

            peaks = peaks[(array_orig[peaks] - M > MAX / 5)]

            if len(peaks) > 0:
                pass
            else:
                break

            allpeaks = numpy.append(allpeaks, peaks)

            for peak in peaks:
                substract += triangle(peak, array[peak], Lambda)[:len(array_orig)]
            array = array - substract
            array = array * (array > 0)


        if len(numpy.atleast_1d(allpeaks)) == 0:
            break

        allpeaks = numpy.sort(allpeaks)
        allpeaks = allpeaks.astype(int)

        dels = []
        for i in range(len(allpeaks)):
            peak = allpeaks[i]

            if peak > 1 and peak < (len(array_orig) - 1):
                if array_orig[peak] < array_orig[peak + 1] or array_orig[peak] < array_orig[peak - 1]:
                    dels.append(i)
        allpeaks = numpy.delete(allpeaks, dels)
        fullpeaklist = numpy.append(fullpeaklist, allpeaks)


    fullpeaklist = numpy.unique(fullpeaklist)
#    fig = plt.plot(array_orig)
#    sc = plt.scatter(fullpeaklist, array_orig[fullpeaklist], color='red')
#    plt.show()
    
    if len(fullpeaklist)>1:
        return fullpeaklist
    else:
        return None









jsonFilePath = '/home/esrf/melnikov/spyder/test/Workflow_20230227-171902/test_meshbest/MeshResults.json'

start_time = time.time()


logger.debug('Checkpoint: Start - {0}s'.format('%0.3f') % (time.time() - start_time))

if os.path.isfile(jsonFilePath):
    json_file = open(jsonFilePath, 'r')
    jsondata = json.load(json_file)
    json_file.close()


Wavelength = jsondata['wavelength']
DetectorDistance = jsondata['detector_distance']
BeamCenter = (jsondata['orgx'], jsondata['orgy'])
DetectorPixel = jsondata['detectorPixelSize']
row, col = jsondata['steps_y'], jsondata['steps_x']



DVH = numpy.frombuffer(base64.b64decode(jsondata['MeshBest']['Cumulative_DVHistogram']))
limits = (0.001, 0.04)
l_values = numpy.linspace(limits[0], limits[1], 100)


peaks = AMPD(DVH)


plt.plot(l_values, DVH)
plt.scatter(l_values[peaks], DVH[peaks], c='red', marker='o')
plt.show()



spots = numpy.frombuffer(base64.b64decode(jsondata['meshPositions'][150]['dozorSpotList']))

spots = spots.reshape((int(spots.size/5), 5))


plt.scatter(spots[:, 1], spots[:, 2], c='blue', marker='+')
plt.show()

RealCoords = numpy.zeros((spots.shape[0], 5))
    
x = (spots[:, 1] - BeamCenter[0]) * DetectorPixel
y = (spots[:, 2] - BeamCenter[1]) * DetectorPixel
divider = Wavelength * numpy.sqrt(x ** 2 + y ** 2 + DetectorDistance ** 2)
RealCoords[:, 0] = x / divider
RealCoords[:, 1] = y / divider
RealCoords[:, 2] = (1/Wavelength) - DetectorDistance/divider



distanceMatrix = spatial.distance.pdist(RealCoords[:, :3], metric='euclidean')
#distanceMatrix = spatial.distance.squareform(distanceMatrix)


print(distanceMatrix.shape)


vectorMatrix = numpy.vstack((spatial.distance.pdist(RealCoords[:, 0].reshape((RealCoords.shape[0], 1))),
                             spatial.distance.pdist(RealCoords[:, 1].reshape((RealCoords.shape[0], 1))),
                             spatial.distance.pdist(RealCoords[:, 2].reshape((RealCoords.shape[0], 1))))).T

print(vectorMatrix.shape)


for peak in l_values[peaks]:

    sigma = 0.05
    print(peak)
    
    
    distanceMask = (distanceMatrix<peak*(1+sigma))*(distanceMatrix>peak*(1-sigma))
    distanceMask = distanceMask.astype(bool)
    
    subset = vectorMatrix[distanceMask]
    
    spherical = Spherical_coords(subset)
    
    
    plt.scatter(spherical[:, 1], spherical[:, 2], c='green', marker='x')
    plt.xlabel('theta')
    plt.ylabel('phi')
    plt.show()


    h = numpy.histogram(spherical[:, 2], bins=100)[0]
    
    plt.plot(h)
    plt.show()






















