#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:14:31 2023

@author: melnikov
"""

import numpy
from scipy.spatial import distance
from matplotlib import pyplot as plt

import logging
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
for h in logger.handlers:
    h.close()
    logger.removeHandler(h)
    
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False



def jacobi_fi(RealCoords, wavelength):
    w = wavelength
    sx, sy, sz = RealCoords.T
    return numpy.array([[-w*sx*sy, sz + w*sz**2 - w*sy**2, -2*w*sy*sz-sy],
                        [-sz - w*sz**2 + w*sx**2, w*sx*sy, 2*w*sx*sz + sx],
                        [w*sy*sz, -w*sx*sz, numpy.zeros(sx.size)]]).T

def calculate_distanceMask(distanceMatrix, threshold=0.5*numpy.pi/180.0):
    def routine(array_with_axis1_shortest):
        logger.debug("distanceMatrix shape: {}".format(array_with_axis1_shortest.shape))
        distanceMask = numpy.zeros(array_with_axis1_shortest.shape)
        nextarray = array_with_axis1_shortest
        mask_axis0 = numpy.ones(array_with_axis1_shortest.shape[0]).astype(bool)
        mask_axis1 = numpy.ones(array_with_axis1_shortest.shape[1]).astype(bool)
        for cycle in range(100):
            mins = numpy.argmin(nextarray, axis=1)
            tempMask = numpy.zeros(nextarray.shape)
            tempMask[numpy.arange(nextarray.shape[0]), mins] = 1
            
            nextarray = numpy.multiply(tempMask, nextarray)
            matches = numpy.min(nextarray, axis=0, where=tempMask.astype(bool), initial=99)
            
            tempMask[nextarray>matches] = 0
            
            distanceMask[numpy.ix_(mask_axis0, mask_axis1)] = tempMask
            
            mask_axis0 = numpy.sum(distanceMask, axis=1)
            mask_axis1 = numpy.sum(distanceMask, axis=0)
            
            if numpy.any(mask_axis0>1) or numpy.any(mask_axis1>1):
                logger.info("Something is not right in matching spots.")
            mask_axis0 = ~mask_axis0.astype(bool)
            mask_axis1 = ~mask_axis1.astype(bool)
            
            nextarray = array_with_axis1_shortest[numpy.ix_(mask_axis0, mask_axis1)]
            logger.debug("Left unmatched spots: {}".format(nextarray.shape))
            if nextarray.size==0:
                break
        return distanceMask
        
    if distanceMatrix.shape[0] > distanceMatrix.shape[1]:
        distanceMask = routine(distanceMatrix.T).T
    else:
        distanceMask = routine(distanceMatrix)
    
    distanceMask[distanceMatrix>threshold] = 0
    
    return distanceMask
        

    


def max_like_rot_axis(dФ, dr_meas):
    matr = numpy.matmul(dФ.transpose((0,2,1)), dФ)
    matr = numpy.sum(matr, axis=0)
#    print(numpy.linalg.det(matr))
    matr = numpy.linalg.inv(matr)

#    print(matr)
    
    second_part = numpy.sum(numpy.matmul(dФ, drs.reshape(drs.shape[0], 3, 1)), axis=0)
    
#    print(matr, second_part)
    result = numpy.matmul(matr, second_part)

    return result#*180/3.14


#    $$$$$$_ $$$$$$$_ _$$$$$__ $$$$$$_
#    __$$___ $$______ $$___$$_ __$$___
#    __$$___ $$$$$___ _$$$____ __$$___
#    __$$___ $$______ ___$$$__ __$$___
#    __$$___ $$______ $$___$$_ __$$___
#    __$$___ $$$$$$$_ _$$$$$__ __$$___ 


def random_shake(spots, sigma_pixels=1.0):
#    numpy.random.seed(1)
    newspots = spots.copy()
    newspots[:, 1] += numpy.random.normal(0.0, sigma_pixels, size=newspots.shape[0])
    newspots[:, 2] += numpy.random.normal(0.0, sigma_pixels, size=newspots.shape[0])
    return newspots

def rotate_z(spots, BeamCenter, fi_deg=0.01):
    newspots = spots.copy()
    newspots[:, 1] -= BeamCenter[0]
    newspots[:, 2] -= BeamCenter[1]

    fi = fi_deg*numpy.pi/180.0
    rot = numpy.array([[numpy.cos(fi), -numpy.sin(fi)],
                        [numpy.sin(fi), numpy.cos(fi)]])
    newspots[:, 1:3] = numpy.dot(rot, newspots[:, 1:3].T).T + numpy.array([BeamCenter[0], BeamCenter[1]])
    
    return newspots


BeamCenter = (1026.16, 1085.48)
Wavelength = 0.96770
DetectorDistance = 114.60
DetectorPixel = 0.075

import time
spots1 = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/00001.spot', skiprows=3)
spots2 = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/00010.spot', skiprows=3)


#spots2 = random_shake(spots1, sigma_pixels=1)



def calculate_dr(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):

    RealCoords1 = numpy.zeros((numpy.shape(spots1)[0], 3))
    RealCoords2 = numpy.zeros((numpy.shape(spots2)[0], 3))
#    plt.scatter(spots1[:, 1], spots1[:, 2], marker='o', c='red', zorder=0, s=0.1)
#    plt.scatter(spots2[:, 1], spots2[:, 2], marker='+', c='blue', lw=0.1)
#    plt.savefig("pat_orient_test.png", dpi=600)
#    plt.show()
    
    
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
    
    distanceMatrix = distance.cdist(RealCoords1, RealCoords2, metric='euclidean')

    threshold = 1.0*numpy.pi/180.0
    
    distanceMask = calculate_distanceMask(distanceMatrix, threshold)


    differenceMatrix = RealCoords2[numpy.newaxis, :] - RealCoords1[:, numpy.newaxis]

    drs = differenceMatrix[distanceMask.astype(bool), :]
    referenceCoords = RealCoords1[numpy.sum(distanceMask, axis=1).astype(bool)]
    
    
    
#    testcoords = spots1[numpy.sum(distanceMask, axis=1).astype(bool)]
#    plt.scatter(spots1[:, 1], spots1[:, 2], marker='o', c='red', s=1e4*distanceMatrix[distanceMask.astype(bool)])
#    plt.show()
    
    
    
    return drs, referenceCoords








#drs, coords = calculate_dr(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)
#
#dФ = jacobi_fi(coords, Wavelength)
#print(dФ.shape)
##print(numpy.linalg.det(dФ))
#
#
#w = max_like_rot_axis(dФ, drs)
#print(w)




#for i in range(8):
#    
##    spots = rotate_z(spots1, BeamCenter, fi_deg=0.01*(i))
##    spots = random_shake(spots1, sigma_pixels=0.5*(i+1))
#    print('/home/esrf/melnikov/spyder/test/LYS_dataset/0000{:.0f}.spot'.format(i+2))
#    spots = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/0000{:.0f}.spot'.format(i+2), skiprows=3)
#    
#    
#
#    drs, coords = calculate_dr(spots1, spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)
#    dФ = jacobi_fi(coords, Wavelength)
#    w = max_like_rot_axis(dФ, drs)
#    print(w)
#    plt.scatter(i, w[1], marker='o', c='blue')
#plt.ylim([-0.001, 0.001])
#plt.show()
#
#
#
#
#
#
#
#
#
#
#calculate_dr(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)






def rotation_estimate_normmove(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):
    RealCoords1 = numpy.zeros((numpy.shape(spots1)[0], 3))
    RealCoords2 = numpy.zeros((numpy.shape(spots2)[0], 3))
#    plt.scatter(spots1[:, 1], spots1[:, 2], marker='o', c='red', zorder=0, s=0.1)
#    plt.scatter(spots2[:, 1], spots2[:, 2], marker='+', c='blue', lw=0.1)
#    plt.savefig("pat_orient_test.png", dpi=600)
#    plt.show()
    
    
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
    
    distanceMatrix = distance.cdist(RealCoords1, RealCoords2, metric='euclidean')

    threshold = 0.5*numpy.pi/180.0
    
    distanceMask = calculate_distanceMask(distanceMatrix, threshold)

    A = numpy.array([[numpy.sum(RealCoords1[:, 1]**2), -numpy.sum(RealCoords1[:, 1]*RealCoords1[:, 0])],
                    [numpy.sum(RealCoords1[:, 1]*RealCoords1[:, 0]), -numpy.sum(RealCoords1[:, 0]**2)]])
    
    spot_width = 0.5
    dr_ = (~numpy.sum(distanceMask, axis=1).astype(bool))*spot_width
    B = numpy.array([numpy.sum(RealCoords1[:, 1]*dr_), numpy.sum(RealCoords1[:, 0]*dr_)])

#    print(A, B)



    axis = numpy.dot(numpy.linalg.inv(A), B)
#    print(axis)

    return axis

#rotation_estimate_normmove(spots1, spots2, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)







for i in range(10):
    
#    spots = rotate_z(spots1, BeamCenter, fi_deg=0.01*(i))
#    spots = random_shake(spots1, sigma_pixels=0.5*(i+1))
    print('/home/esrf/melnikov/spyder/test/LYS_dataset/000{:02d}.spot'.format(i+2))
    spots = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/000{:02d}.spot'.format(i+2), skiprows=3)
    
    

    w = rotation_estimate_normmove(spots1, spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)
    print(w)
    plt.scatter(i, w[0], marker='o', c='blue')
    plt.scatter(i, w[1], marker='o', c='red')
#plt.ylim([-0.001, 0.001])
plt.show()

























