#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:12:05 2024

@author: melnikov
"""

__version__ = 1.0

import os
import numpy
import base64
import ctypes
import json
import glob
import time
from matplotlib import pyplot as plt
from meshbest import rings, dvanalysis
from doublemesh import lattice_vector_search

try:
    import billiard as mp
except ModuleNotFoundError:
    import multiprocessing as mp


stddc = False
path = '/data/id23eh1/inhouse/opid231/20240208/PROCESSED_DATA/LYS/LYS-Lys_03/run_01_MXPressE/'

#path = '/data/id23eh1/inhouse/opid231/20240301/PROCESSED_DATA/2MESH_TEST_OA14/Sample-3:1:04/run_01_MXPressF/'

#path = '/data/id23eh1/inhouse/opid231/20240213/PROCESSED_DATA/Tryp/Tryp-Tryp01/run_03_MXPressF/'

#path = '/data/id23eh1/inhouse/opid231/20240301/PROCESSED_DATA/2MESH_TEST_OA14/Sample-1:2:10/run_01_MXPressF/'
    
#path, stddc = '/data/visitor/mx2607/id30a3/20240309/PROCESSED_DATA/JP-Lab/scTrl1/scTrl1-SK202/', True

DO_PLOT = True
DSCORE_THRESHOLD = 0.2

try:
    from bes.workflow_lib import workflow_logging

    logger = workflow_logging.getLogger()
    DO_PLOT = False

except Exception:
    import logging

    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)


def createjson(path, stddc=False):
#    st = time.time()
    if stddc:
        temp = glob.glob(path+'autoprocessing_'+path.split('/')[-2]+'*/dozor/ControlDozor_*')[0]
    else:
        temp = glob.glob(path+'run_*_datacollection/autoprocessing_*_1/dozor/nobackup/ImageQualityIndicators_????????')[0]
#    print(temp)
    
    pixelsize = 0.075
    
    inmeta = glob.glob(temp+'/ControlDozor_*/ExecDozor_*/inDataExecDozor.json')[0]
    
    with open(inmeta, 'r') as j:
        JSON = json.load(j)
        j.close()
#    print('Checkpoint1: {:.2f} s'.format((time.time()-st)))
    JSON['detectorPixelSize'] = pixelsize
    
    if stddc:
        with open(temp+'/outDataControlDozor.json', 'r') as j:
            JSON2 = json.load(j)
            j.close()
    else:
        with open(temp+'/outDataImageQualityIndicators.json', 'r') as j:
            JSON2 = json.load(j)
            j.close()
    
    JSON['imageQualityIndicators'] = JSON2['imageQualityIndicators']
#    print('Checkpoint2: {:.2f} s'.format((time.time()-st)))
    size = len(JSON['imageQualityIndicators'])
    
    for i in range(size):
        if 'dozorSpotFile' in JSON['imageQualityIndicators'][i].keys():
            path = JSON['imageQualityIndicators'][i]['dozorSpotFile']
#            print(JSON['imageQualityIndicators'][i].keys())
#            print(path)
            if os.path.isfile(path):
                array = numpy.loadtxt(path, skiprows=3)
                text = base64.b64encode(array).decode()
                JSON['imageQualityIndicators'][i][u'dozorSpotList'] = text

    return JSON
#------------------------------------------------------------------------------

def percentileRange(array, level=10, roll_w=50, acceptance=0.9):
    window = numpy.ones(roll_w)
    bestspot = 0
    for _ in range(1000):
        thr = numpy.percentile(array, level)
        success = (array<thr).astype(int)
        conv = numpy.convolve(success, window)/roll_w
        if numpy.any(conv>acceptance):
            plt.plot(conv)
            plt.show()
            bestspot = numpy.argmax(conv)
            bestspot = bestspot - roll_w//2
#            print(success[bestspot:bestspot+roll_w].sum())
            break
        else:
            level += 5
    return bestspot

def selection_cycle_indexing(jsondata, required_angle=10, number_of_regions=2):
    dscore = numpy.copy(numpy.frombuffer(base64.b64decode(jsondata['dozorScoreArray'])))
    mcdarray = numpy.copy(numpy.frombuffer(base64.b64decode(jsondata['MCDArray'])))
    mask = dscore>DSCORE_THRESHOLD
    mcdarray[~mask] = numpy.inf
    
    angle_increment = jsondata['oscillationRange']
    roll_w = int(required_angle/angle_increment)
    window = numpy.ones(roll_w)

    conv = numpy.convolve(mcdarray, window, 'valid')/roll_w
#    plt.plot(mcdarray)
#    plt.show()
#    plt.plot(conv)
#    plt.show()
    
    if number_of_regions==2:
        matr = conv.reshape(conv.size, -1) + conv.reshape(conv.size, -1).T
#        plt.imshow(matr, cmap='hot')
#        plt.colorbar()
#        plt.show()
        
        def distance_factor(size, angle_increment):
            angles = angle_increment*numpy.arange(size)
            angles = numpy.abs(angles.reshape(size, -1) - angles.reshape(size, -1).T)
            sigma = 10.0
            l = numpy.zeros((size, size))
            for f in [0.0, 180.0, 360.0]:
                l += numpy.exp(-(angles-f)**2/(2*(sigma**2)))
            l = 3*l + 1
            return l
    
        
        d_factor = distance_factor(conv.size, angle_increment)
#        plt.imshow(d_factor)
#        plt.colorbar()
#        plt.show()
        
        matr = numpy.multiply(matr, d_factor)
#        plt.imshow(numpy.log(matr), cmap='hot')
#        plt.colorbar()
#        plt.show()
        
        regions = numpy.argwhere(matr==matr.min())[0]
        return regions, numpy.mean(conv[regions])
    elif number_of_regions==1:
        regions = numpy.argmin(conv)
        return regions, conv[regions]
    else:
#        logger.error('Wrong number of regions set; use either 1 or 2')
        pass

def construct3Dpositions(jsondata, starting_frame, ending_frame):
    
    RealCoords = numpy.empty(shape=(0, 3))
        
    Wavelength = jsondata['wavelength']
    DetectorDistance = jsondata['detectorDistance']
    BeamCenter = (jsondata['orgx'], jsondata['orgy'])
    DetectorPixel = jsondata['detectorPixelSize']
    
    for image in range(starting_frame, ending_frame, 5):
        if 'dozorSpotList_norings' in jsondata['imageQualityIndicators'][image].keys():
            spots = dvanalysis.from64ToSpotArray(jsondata['imageQualityIndicators'][image]['dozorSpotList_norings'])
            
            array3d = numpy.zeros((spots.shape[0], 3))
            x = (spots[:, 1] - BeamCenter[0]) * DetectorPixel
            y = (spots[:, 2] - BeamCenter[1]) * DetectorPixel
            divider = Wavelength * numpy.sqrt(x**2 + y**2 + DetectorDistance**2)
            array3d[:, 0] = x / divider
            array3d[:, 1] = y / divider
            array3d[:, 2] = (1 / Wavelength) - DetectorDistance / divider
    
            current_angle_delta = (numpy.pi/180.0)*jsondata['oscillationRange']*(image - starting_frame)
            
#            print(array3d)
            rotated = lattice_vector_search.rotate_vector(array3d.T, numpy.array([1.0, 0.0, 0.0]), current_angle_delta).T
#            print(rotated)
            RealCoords = numpy.vstack((RealCoords, rotated))
        
    return RealCoords

def checkLatticeMisorientation(jsondata, spot_range1, spot_range2=None):

    Wavelength = jsondata['wavelength']
    DetectorDistance = jsondata['detectorDistance']
    BeamCenter = (jsondata['orgx'], jsondata['orgy'])
    DetectorPixel = jsondata['detectorPixelSize']


    if spot_range2!=None:
        frame1 = spot_range1[0]
        RealCoords1 = construct3Dpositions(jsondata, spot_range1[0], spot_range1[1])
        frame2 = spot_range2[0]
        RealCoords2 = construct3Dpositions(jsondata, spot_range2[0], spot_range2[1])
    else:
        frame1 = spot_range1[0]
        RealCoords1 = construct3Dpositions(jsondata, spot_range1[0], spot_range1[0]+1)
        frame2 = spot_range1[1]
        RealCoords2 = construct3Dpositions(jsondata, spot_range1[1], spot_range1[1]+1)
    
    angle_delta12 = (numpy.pi/180.0)*jsondata['oscillationRange']*(frame2 - frame1)
    
    check = lattice_vector_search.crosscheck2patterns(RealCoords1, RealCoords2, angle_delta12, BeamCenter,
                                                      Wavelength, DetectorDistance, DetectorPixel, spots_in3D=True)
    
    return check

def quickrunMCD(jsondata):
    start = time.time()
    logger.info('Starting MCD')
    rings.removeSaltRings(jsondata)
    dvanalysis.determineMCD(jsondata)
    dscore = [item['dozorScore'] for item in jsondata['imageQualityIndicators']]
    dscore = numpy.array(dscore).astype(float)
    jsondata['dozorScoreArray'] = base64.b64encode(dscore).decode()
    logger.info('MCD analysis - Elapsed: {:.2f} s'.format((time.time()-start)))

def bestPartForIndexing(jsondata, doLatticeCheck=False):
    if not 'MCDArray' in jsondata.keys():
        quickrunMCD(jsondata)

    '''trying first to get good 2x10-degree range'''
    a = 10.0
    regions, mcd_quality = selection_cycle_indexing(jsondata, required_angle=a, number_of_regions=2)
    if mcd_quality>2e-4:
        '''if not, trying to get good 2x5-degree range'''
        a = 5.0
        regions, mcd_quality = selection_cycle_indexing(jsondata, required_angle=a, number_of_regions=2)
        if mcd_quality>2e-4:
            '''else, trying to get good 1x10-degree range'''
            a = 10.0
            regions, mcd_quality = selection_cycle_indexing(jsondata, required_angle=a, number_of_regions=1)
    
    angle_increment = jsondata['oscillationRange']
    roll_w = int(a/angle_increment)
#    print(roll_w, angle_increment)
    result = [[i, i+roll_w] for i in regions]
    
    if doLatticeCheck:
        check = checkLatticeMisorientation(jsondata, result[0], spot_range2=(None if len(result)<2 else result[1]))
        if check[0]<0.5 and check[1]<2:
            logger.info('Detected misorientation in lattice over the data set!')
    return result

def bestPartForIntegration(jsondata, cutoff=2e-4, minosc=20, doLatticeCheck=False):
    if not 'MCDArray' in jsondata.keys():
        quickrunMCD(jsondata)

    dscore = numpy.copy(numpy.frombuffer(base64.b64decode(jsondata['dozorScoreArray'])))
    mcdarray = numpy.copy(numpy.frombuffer(base64.b64decode(jsondata['MCDArray'])))
    mask = dscore>DSCORE_THRESHOLD
    mcdarray[~mask] = numpy.inf
    if mcdarray.mean()<=cutoff:
        if doLatticeCheck:
            check = checkLatticeMisorientation(jsondata, [0, mcdarray.size])
            if check[0]<0.5 and check[1]<2:
                logger.info('Detected misorientation in lattice over the data set!')
        return [0, mcdarray.size]
    else:
        L = int(mcdarray.size)-3
        a2d = numpy.repeat([mcdarray], L, axis=0)
        result = numpy.ones(a2d.shape)
        for i in range(L):
            result[i][:mcdarray.size-i-1] = numpy.convolve(a2d[i], numpy.ones(i+2), mode='valid')/(i+2)
#        plt.imshow(result)
#        plt.colorbar()
#        plt.show()
        
        width, start = numpy.where(result<=cutoff)
        if width[-1]*jsondata['oscillationRange']<minosc:
            width, start = numpy.where(result<=cutoff*1.25)
        
        if doLatticeCheck:
            check = checkLatticeMisorientation(jsondata, [0, mcdarray.size])
            if check[0]<0.5 and check[1]<2:
                logger.info('Detected misorientation in lattice over the data set!')
        
        return [start[-1], start[-1]+width[-1]]

def selftest():
    numpy.random.seed(1)
    
    start = time.time()
    
    j = createjson(path, stddc=stddc)
    
    logger.info('Json loading - Elapsed: {:.2f} s'.format((time.time()-start)))
    start = time.time()
    logger.info('Starting MCD')
    
    rings.removeSaltRings(j)
    
    dvanalysis.determineMCD(j)
    
    dscore = [item['dozorScore'] for item in j['imageQualityIndicators']]
    dscore = numpy.array(dscore).astype(float)
    j['dozorScoreArray'] = base64.b64encode(dscore).decode()
    
    logger.info('MCD analysis - Elapsed: {:.2f} s'.format((time.time()-start)))
    start = time.time()
    
    n = bestPartForIndexing(j,
#                            doLatticeCheck=True
                            )
    logger.info('Useful range for indexing: ' + '; '.join(['-'.join(i.astype(str)) for i in numpy.array(n)]))
    logger.info('Autorange for indexing intervals - Elapsed: {:.2f} s'.format((time.time()-start)))
    start = time.time()
    
    f = bestPartForIntegration(j)
    logger.info(f)
    logger.info('Autorange for integration intervals - Elapsed: {:.2f} s'.format((time.time()-start)))
    
#    plt.plot(dscore, c='indigo')
#    for i in n:
#        plt.axvspan(i[0], i[1], alpha=0.5, color='red')
#    plt.title('dozor score')
#    plt.show()
    
    mcdarray = numpy.copy(numpy.frombuffer(base64.b64decode(j['MCDArray'])))
    plt.plot(mcdarray)
    for i in n:
        plt.axvspan(i[0], i[1], alpha=0.5, color='red')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title('mcd_score')
    plt.show()

#def check_MPwrapper(queue):
#    global Buffer
#    while True:
#        spots3d, i, v0 = queue.get()
#        if not isinstance(spots3d, numpy.ndarray):
#            break
#        
#        xi = numpy.unravel_index(i, (21, 21))
#        q = 0.1
#        phi, th = numpy.meshgrid(numpy.linspace(-q, q, 21), numpy.linspace(-q, q, 21))
#
#        sph = lattice_vector_search.xyzToSpherical(v0)
#        sph[1] = sph[1] + th[xi]
#        sph[2] = sph[2] + phi[xi]
#        v_ = lattice_vector_search.sphericalToXYZ(sph)
#        
#        check = lattice_vector_search.check(spots3d, v_)
#        
#        Buffer[i] = check
#
#def orientationReliability(jsondata, reference_spot_range=None):
#    ref = reference_spot_range if reference_spot_range!=None else [0, 10]
#    Wavelength = jsondata['wavelength']
#    DetectorDistance = jsondata['detectorDistance']
#    BeamCenter = (jsondata['orgx'], jsondata['orgy'])
#    DetectorPixel = jsondata['detectorPixelSize']
#    N = len(jsondata['imageQualityIndicators'])
#    
#    RealCoords = construct3Dpositions(jsondata, ref[0], ref[1])
#    v, _ = lattice_vector_search.find_planes(
#        spots=RealCoords,
#        BeamCenter=BeamCenter,
#        Wavelength=Wavelength,
#        DetectorDistance=DetectorDistance,
#        DetectorPixel=DetectorPixel,
#        spots_in3D=True
#        )
#    
#    #Keeping only good vectors
#
#    v = numpy.array(v)
#    v = v[v[:, 3]>3]
#    
#    global Buffer
#    Buffer = mp.RawArray(ctypes.c_double, 21*21)
#
#    nCPU = mp.cpu_count()
#    logger.info("Starting multicore processing; CPU count: {}".format(nCPU))
#    queue = mp.Queue()
#  
##    for frame in range(0, N, 50):
##        angle_delta = (numpy.pi/180.0)*jsondata['oscillationRange']*(frame - ref[0])+3*numpy.random.random()
##        if 'dozorSpotList_norings' in jsondata['imageQualityIndicators'][frame].keys():
##            spots3d = construct3Dpositions(jsondata, frame, frame+20)
###            spots = dvanalysis.from64ToSpotArray(jsondata['imageQualityIndicators'][frame]['dozorSpotList_norings'])
##        else:
##            continue
##
##        queue.put((spots3d, frame, v, angle_delta))
#
#    v0 = v[0]
#    for i in range(21*21):
#        queue.put((RealCoords, i, v0))
#
#    for item in range(nCPU):
#        queue.put((None, None, None))
#
#    workers = []
#    for item in range(nCPU):
#        worker = mp.Process(
#            target=check_MPwrapper,
#            args=(
#                queue,
##                BeamCenter,
##                Wavelength,
##                DetectorDistance,
##                DetectorPixel,
#            ),
#        )
#        workers.append(worker)
#        worker.start()
#    for worker in workers:
#        worker.join()
#    
#    check_map = numpy.frombuffer(Buffer).astype(float).reshape((21, 21))
#    
#    plt.imshow(check_map, cmap='hot')
#    plt.colorbar()
#    plt.show()







#selftest()






