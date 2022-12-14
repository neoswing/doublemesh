#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:31:42 2021

@author: melnikov
"""

import os
import re
import glob
import time
import multiprocessing as mp
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy
import lattice_vector_search

try:
    from workflow_lib import workflow_logging
    logger = workflow_logging.getLogger()
except:
    import logging
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)



def extractDozorMetadata(datfilename):
    with open(datfilename, 'r') as metadata:
        lines = metadata.readlines()
    metadata.close()
    
    orgx, orgy, Wavelength, DetectorDistance, DetectorPixel, Phi1, Phi2  = None, None, None, None, None, None, None
    
    for line in lines:
        if re.match('orgx', line):
            orgx = float(line.split(' ')[-1])
        if re.match('orgy', line):
            orgy = float(line.split(' ')[-1])
        if re.match('X-ray_wavelength', line):
            Wavelength = float(line.split(' ')[-1])
        if re.match('detector_distance', line):
            DetectorDistance = float(line.split(' ')[-1])
        if re.match('pixel', line):
            DetectorPixel = float(line.split(' ')[-1])
        if re.match('phi1', line):
            Phi1 = float(line.split(' ')[-1])
        if re.match('phi2', line):
            Phi2 = float(line.split(' ')[-1])
    
    if isinstance(orgx, float) and isinstance(orgy, float)\
    and isinstance(Wavelength, float) and isinstance(DetectorDistance, float)\
    and isinstance(DetectorPixel, float) and isinstance(Phi1, float) and isinstance(Phi2, float):
        angle_delta = Phi2 - Phi1
        return (orgx, orgy), Wavelength, DetectorDistance, DetectorPixel, angle_delta
    else:
        logger.error('Problem in reading dat file')
        return
    
def findPlanes_MP(queue, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):
    global Buffer
    while True:
        spots, crystal_n = queue.get()
#        print(spots)
        if not isinstance(spots, numpy.ndarray):
            break

        v, RealCoords = lattice_vector_search.find_planes(spots,
                                                          BeamCenter=BeamCenter,
                                                          Wavelength=Wavelength,
                                                          DetectorDistance=DetectorDistance,
                                                          DetectorPixel=DetectorPixel)

        Buffer[crystal_n] = v, RealCoords


def analyseDoubleMeshscan(path):
    global Buffer
    initialCWD = os.getcwd()
    os.chdir(path)

    crystals1 = glob.glob('crystal_1_*.spot')
    crystals1.sort()
    crystals2 = glob.glob('crystal_2_*.spot')
    crystals2.sort()
    
    crystals_mesh1 = [numpy.loadtxt(name, skiprows=1, dtype=float) for name in crystals1]
    crystals_mesh2 = [numpy.loadtxt(name, skiprows=1, dtype=float) for name in crystals2]

    
    
    BeamCenter, Wavelength, DetectorDistance, DetectorPixel, angle_delta = extractDozorMetadata(glob.glob('dozorm2.dat')[0])
    
    
    logger.debug('Experiment metadata: BeamCenter {0} {1}, Wavelength {2}, DtoX {3}, Pixelsize {4}, Omega difference {5}'.format(BeamCenter[0],
                                                                                                                                 BeamCenter[1],
                                                                                                                                 Wavelength,
                                                                                                                                 DetectorDistance,
                                                                                                                                 DetectorPixel,
                                                                                                                                 angle_delta))
    
    angle_delta = angle_delta*3.14/180.0
    
    potentialMatches = numpy.loadtxt('dozorm_pair.dat')
    potentialMatches = numpy.hstack((potentialMatches, numpy.zeros((potentialMatches.shape[0], 1))))

    manager = mp.Manager()
    Buffer = manager.dict()
    nCPU = mp.cpu_count()

    queue = mp.Queue()    
    
    
    
    i = 0
    for spots in crystals_mesh1:
        spots = numpy.hstack((numpy.zeros((spots.shape[0], 1)), spots))
        spots = spots[spots[:, -1]>spots[:, -1].max()/2.0]
        print(spots.shape)
        queue.put((spots, i))
        i += 1
    for item in range(nCPU):
        queue.put((None, None))

    workers = []
    for item in range(nCPU):
        worker = mp.Process(target=findPlanes_MP, args=(queue, BeamCenter, Wavelength, DetectorDistance, DetectorPixel,))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    
    Buffer0 = Buffer
    Buffer = manager.dict()

    i = 0
    for spots in crystals_mesh2:
        spots = numpy.hstack((numpy.zeros((spots.shape[0], 1)), spots))
        spots = spots[spots[:, -1]>spots[:, -1].max()/2.0]
        print(spots.shape)
        queue.put((spots, i))
        i += 1
    for item in range(nCPU):
        queue.put((None, None))

    workers = []
    for item in range(nCPU):
        worker = mp.Process(target=findPlanes_MP, args=(queue, BeamCenter, Wavelength, DetectorDistance, DetectorPixel,))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    



    for line in potentialMatches:
        matches = []
        conf = []
        
        i = line[1] - 1
        j = line[2] - 1
        
        vectorsi = Buffer0[i][0]
        vectorsj = Buffer[j][0]
        
        RealCoordsi = Buffer0[i][1]
        RealCoordsj = Buffer[j][1]
        
        for vi in vectorsi:
            newvi = lattice_vector_search.rotate_vector(vi[:3], numpy.array([1.0, 0.0, 0.0]), -angle_delta)
            newvi = numpy.hstack((newvi, vi[3]))
            chck = lattice_vector_search.check(RealCoordsj, newvi)
            matches.append(chck)
            conf.append(newvi[3])
        for vj in vectorsj:
            newvj = lattice_vector_search.rotate_vector(vj[:3], numpy.array([1.0, 0.0, 0.0]), angle_delta)
            newvj = numpy.hstack((newvj, vj[3]))
            chck = lattice_vector_search.check(RealCoordsi, newvj)
            matches.append(chck)
            conf.append(newvj[3])
        
        matches = numpy.asarray(matches)
        conf = numpy.asarray(conf)
        line[3] = lattice_vector_search.sigmoid(numpy.sum(matches*numpy.exp(conf))/numpy.exp(conf).sum())

    potentialMatches = numpy.hstack((potentialMatches, (potentialMatches[:, 3]>0.5).reshape(potentialMatches.shape[0], 1)))
    
    print("Success")
    print("Case#| Xtal1 | Xtal2 | Score | Y/N")
    for item in potentialMatches:
    	print("{0:2.0f}   |  {1:2.0f}   |  {2:2.0f}   | {3:1.2f}  | {4:1.0f}".format(item[0], item[1], item[2], item[3], item[4]))
    
    numpy.savetxt('dozorm_pair_final.dat', potentialMatches, fmt='%d %d %d %.2f %d')
    plt.close()




#    crosscheck_matrix = numpy.zeros((len(crystals_mesh1), len(crystals_mesh2)))
#
#    for i, j in numpy.ndindex(crosscheck_matrix.shape):
#        matches = []
#        conf = []
#
#        vectorsi = Buffer0[i][0]
#        vectorsj = Buffer[j][0]
#        
#        RealCoordsi = Buffer0[i][1]
#        RealCoordsj = Buffer[j][1]
#        
#        for vi in vectorsi:
#            newvi = lattice_vector_search.rotate_vector(vi[:3], numpy.array([1.0, 0.0, 0.0]), -angle_delta)
#            newvi = numpy.hstack((newvi, vi[3]))
#            chck = lattice_vector_search.check(RealCoordsj, newvi)
#            matches.append(chck)
#            conf.append(newvi[3])
#        for vj in vectorsj:
#            newvj = lattice_vector_search.rotate_vector(vj[:3], numpy.array([1.0, 0.0, 0.0]), angle_delta)
#            newvj = numpy.hstack((newvj, vj[3]))
#            chck = lattice_vector_search.check(RealCoordsi, newvj)
#            matches.append(chck)
#            conf.append(newvj[3])
#        
#        matches = numpy.asarray(matches)
#        conf = numpy.asarray(conf)
#        crosscheck_matrix[i, j] = lattice_vector_search.sigmoid(numpy.sum(matches*numpy.exp(conf))/numpy.exp(conf).sum())
#
#    plt.close()
#    plt.imshow(crosscheck_matrix, cmap='hot', origin='bottom left',
#               extent=[0.5, crosscheck_matrix.shape[1]+0.5, 0.5, crosscheck_matrix.shape[0]+0.5])
#    plt.xticks(numpy.arange(1, crosscheck_matrix.shape[1]+1, 2))
#    plt.yticks(numpy.arange(1, crosscheck_matrix.shape[0]+1, 2))
#    plt.colorbar()
#    plt.savefig('Crosscheck_matrix.png', dpi=300)
#    plt.close()
    
    
    
    
    
    
    
    os.chdir(initialCWD)
















start = time.time()



#analyseDoubleMeshscan('/home/esrf/melnikov/test_double_meshscan')
#
##1st crystal 1st meshscan directions:
##[array([-19.99, -63.07, 749.17, 1.50]), array([-8.05, 83.53, 440.42, 2.52]),
##array([59.83, 64.93, 196.65, 1.11]), array([69.92, 63.79, 190.26, 2.05])]










os.chdir('/home/esrf/melnikov/test_double_meshscan')
crystals1 = glob.glob('crystal_1_*.spot')
crystals1.sort()
crystals2 = glob.glob('crystal_2_*.spot')
crystals2.sort()

crystals_mesh1 = [numpy.loadtxt(name, skiprows=1, dtype=float) for name in crystals1]
crystals_mesh2 = [numpy.loadtxt(name, skiprows=1, dtype=float) for name in crystals2]

BeamCenter, Wavelength, DetectorDistance, DetectorPixel, angle_delta = extractDozorMetadata(glob.glob('d.dat')[0])


i = 0
for spots in crystals_mesh1[0:3]:
    i += 1
    print('Crystal {} <-----------------------------'.format(i))
    spots = numpy.hstack((numpy.zeros((spots.shape[0], 1)), spots))
    
    plt.scatter(spots[:, 1], spots[:, 2], marker='+', c='blue')
    plt.show()
    
    spots = spots[spots[:, -1]>spots[:, -1].max()/2]
    plt.scatter(spots[:, 1], spots[:, 2], marker='o', c='red')
    plt.show()
    print(spots.shape)
    v, RealCoords = lattice_vector_search.find_planes(spots,
                                                      BeamCenter=BeamCenter,
                                                      Wavelength=Wavelength,
                                                      DetectorDistance=DetectorDistance,
                                                      DetectorPixel=DetectorPixel)









logger.info('Elapsed: {:.2f}s'.format(time.time()-start))