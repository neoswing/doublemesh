# -*- coding: utf-8 -*-
"""
By Igor Melnikov

04/08/2021
"""

__version__ = 2.1
'v.2.1 fixed minor bug'

import numpy
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance
from matplotlib import pyplot as plt


import logging
logger = logging.getLogger("test")
logger.setLevel(logging.INFO)
for h in logger.handlers:
    h.close()
    logger.removeHandler(h)
    
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False




float_formatter = "{:.2f}".format
numpy.set_printoptions(formatter={'float_kind':float_formatter})

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def sigmoid(x, a=4.39, x0=1.0):
    return 1/(1+numpy.exp(-a*(x-x0)))

def Spherical_coords(xyz):
    ptsnew = numpy.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = numpy.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = numpy.arctan2(numpy.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = numpy.arctan2(xyz[:,2], numpy.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = numpy.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def backToXYZ(Sphcoords):
    xyz = numpy.zeros(Sphcoords.shape)
    xyz[2] = Sphcoords[0]*numpy.cos(Sphcoords[1])
    xyz[0] = Sphcoords[0]*numpy.sin(Sphcoords[1])*numpy.cos(Sphcoords[2])
    xyz[1] = Sphcoords[0]*numpy.sin(Sphcoords[1])*numpy.sin(Sphcoords[2])
    return xyz

def density(x, sigma, points_coords):
    p = points_coords[:, numpy.newaxis] - x
    return numpy.sum(numpy.exp(-(p**2)/(2*sigma**2)), axis=0)

def direction_proj(newaxR, RealCoords):
    return numpy.sum(numpy.multiply(newaxR, RealCoords), axis=1)

def decimal(x):
    a1 = x%1
    a2 = (1-x)%1
    return numpy.min(numpy.abs([a1, a2]), axis=0)

def rotate_vector(vector, axis, angle):
#    logger.debug('Axis: {}'.format(axis))
    axis = axis/numpy.linalg.norm(axis)
    s = numpy.sin(angle)
    c = numpy.cos(angle)
    cp_matr = numpy.array([[0, -axis[2], axis[1]],
                           [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])

    rotation_matrix = numpy.eye(3) + s*cp_matr + (1-c)*numpy.dot(cp_matr, cp_matr)

#    logger.debug('Rotation matrix: {}'.format(rotation_matrix))
    result = numpy.dot(rotation_matrix, vector)
    return result

def find_peaks2D(array2D, onepeak=False):
    if onepeak:
        array2D = ndimage.gaussian_filter(array2D, sigma=2)
#        plt.imshow(array2D, cmap='hot', origin="lower")
#        plt.show()
        return [numpy.unravel_index(numpy.argmax(array2D), array2D.shape)]
    else:
        peak_indices = []
        array2D -= ndimage.gaussian_filter(array2D, sigma=5)
#        plt.imshow(array2D, cmap='hot')
#        plt.imshow(((array2D/(array2D.mean()+3*array2D.std()))>1.0), cmap='hot')
#        plt.show()
        zones = ndimage.measurements.label((array2D>array2D.mean()+3*array2D.std()), structure=numpy.ones((3, 3)))
#        zones = ndimage.measurements.label((array2D>0.3), structure=numpy.ones((3, 3)))
        for i in range(zones[1]):
            zone = numpy.where(zones[0]==i+1)
            center = numpy.argmax(array2D[zone])
            center = zone[0][center], zone[1][center]
            peak_indices.append(center)
        
        return peak_indices

#def refine_peak(phi_theta, RealCoords)

def alignment_score(dr, RealCoords):
    dr = backToXYZ(numpy.array([1, dr[0], dr[1]]))
    newdr_coords = numpy.dot(dr, RealCoords.T)
    histtt = numpy.histogram(newdr_coords, bins=100)[0]
    FT0 = histtt.sum()
    histtt -= ndimage.gaussian_filter1d(histtt, sigma=10)
    ft = numpy.abs(numpy.fft.rfft(histtt))
    
    return numpy.max(ft)/FT0
    


def find_planes(spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):
    ''' Returns plane normale vector XYZ with interplane frequency as length and fourier peak height'''
#    st = time.time()
    RealCoords = numpy.zeros((numpy.shape(spots)[0], 3))

    x = (spots[:, 1] - BeamCenter[0]) * DetectorPixel
    y = (spots[:, 2] - BeamCenter[1]) * DetectorPixel
    divider = Wavelength * numpy.sqrt(x ** 2 + y ** 2 + DetectorDistance ** 2)
    RealCoords[:, 0] = x / divider
    RealCoords[:, 1] = y / divider
    RealCoords[:, 2] = (1/Wavelength) - DetectorDistance/divider
#    RealCoords[i, 3] = spots[i, 0]
#    RealCoords[i, 4] = float(spots[i, 3]) / float(spots[i, 4])

    if len(numpy.atleast_1d(spots)) < 50:
        logger.error('Not enough spots for proper analysis in some of the crystals')
        return [], RealCoords
    else:

        bining = 50
        phis = numpy.linspace(0, 3.14, bining)
        thetas = numpy.linspace(0, 3.14, bining)

        Z = numpy.zeros((bining, bining))
        for i, j in numpy.ndindex((bining, bining)):
            Z[i, j] = alignment_score((thetas[i], phis[j]), RealCoords)
        plt.imshow(Z, cmap='hot', interpolation='nearest', origin='lower', extent=[phis.min(), phis.max(), thetas.min(), thetas.max()])
        plt.colorbar()
        plt.show()
        logger.debug('Peak areas: {0}'.format(numpy.where(Z>Z.mean()+10*Z.std())))
                
        peaks = find_peaks2D(Z)
        logger.debug('Peaks: {0}'.format(peaks))
        bining = 50
        vectors = []
#        print("Main part finished in {} s".format((time.time()-st)))
        for peak in peaks:
            maindirection = thetas[peak[0]], phis[peak[1]]
            logger.debug('Main direction unrefined: phi={0:.2f}, theta={1:.2f}'.format(maindirection[1], maindirection[0]))
            bining2 = 11
            boundary = 0.05
            
            p = numpy.linspace(maindirection[1]-boundary, maindirection[1]+boundary, bining2)
            t = numpy.linspace(maindirection[0]-boundary, maindirection[0]+boundary, bining2)

            Z = numpy.zeros((bining2, bining2))
            for i, j in numpy.ndindex((bining2, bining2)):
                Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
            plt.imshow(Z, cmap='hot', interpolation='nearest', origin='lower', extent=[p.min(), p.max(), t.min(), t.max()])
            plt.colorbar()
            plt.show()
            refine_peak = find_peaks2D(Z, onepeak=True)
            logger.debug('Refined peak: {}'.format(refine_peak))
            
            maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
            #iteration
            logger.debug('Main direction unrefined: phi={0:.2f}, theta={1:.2f}'.format(maindirection[1], maindirection[0]))
            
            boundary = 0.01
            
            p = numpy.linspace(maindirection[1]-boundary, maindirection[1]+boundary, bining2)
            t = numpy.linspace(maindirection[0]-boundary, maindirection[0]+boundary, bining2)

            Z = numpy.zeros((bining2, bining2))
            for i, j in numpy.ndindex((bining2, bining2)):
                Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
            plt.imshow(Z, cmap='hot', interpolation='nearest', origin='lower', extent=[p.min(), p.max(), t.min(), t.max()])
            plt.colorbar()
            plt.show()
            refine_peak = find_peaks2D(Z, onepeak=True)
            logger.debug('Refined peak: {}'.format(refine_peak))
            
            maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
            #iteration fin
            logger.debug('Main direction refined: phi={0:.2f}, theta={1:.2f}'.format(maindirection[1], maindirection[0]))

            maindirectionXYZ = backToXYZ(numpy.array([1.0, maindirection[0], maindirection[1]]))
            proj_coords = direction_proj(maindirectionXYZ, RealCoords)
            
            bins = int(0.8*proj_coords.size)
            bins = bins if bins>=50 else 50
            dens = numpy.histogram(proj_coords, bins=bins)[0].astype(float)
            plt.plot(dens)
            plt.show()
            dens -= ndimage.gaussian_filter1d(dens, sigma=0.05*proj_coords.size)
            dens -= numpy.mean(dens)
            fourier = numpy.abs(numpy.fft.rfft(dens))
            plt.plot(fourier)
            plt.show()
            
            peak = int(numpy.argmax(fourier))
            w = numpy.pi*peak/(proj_coords.max()-proj_coords.min())
            noise = fourier[numpy.delete(numpy.arange(fourier.size), numpy.arange(fourier.size)[::peak])]
            score = (fourier[peak] - noise.mean()) / (5*noise.std())
            
            logger.debug('Score for direction: {:.2f}'.format(score))
            if score>=1:
                vectors.append(numpy.hstack((w*maindirectionXYZ, score)))
            else:
                logger.info('Score is too low, aborting this direction: {:.2f}'.format(score))
        return vectors, RealCoords

def check(RealCoords, plane_vector):
    freq = numpy.linalg.norm(plane_vector[:3])
    if freq>300:
        logger.info('Detected large cell parameter ({0:.2f}'.format(freq)+u'\u212B'+
                    '): vector {0} with confidence score {1:.2f}'.format(plane_vector[:3], plane_vector[3]))
        logger.info('No trust to this one')
    direction = plane_vector[:3]/freq
    proj_coords = direction_proj(direction, RealCoords)
    n_peak = int(freq*(proj_coords.max()-proj_coords.min())/numpy.pi)
    n_peak = n_peak if n_peak>0 else 1
    
    bins = int(max(0.04*proj_coords.size, 100, 4*n_peak))
    logger.debug('Number of bins: {}'.format(bins))

    dens = numpy.histogram(proj_coords, bins=bins)[0].astype(float)
    plt.plot(dens)
    plt.show()
    dens -= ndimage.gaussian_filter1d(dens, sigma=0.05*proj_coords.size)
    dens -= numpy.mean(dens)
    fourier = numpy.abs(numpy.fft.rfft(dens))

    plt.plot(fourier)
    plt.show()

    logger.info('Search for peak in: {0}, corresponding to frequency {1:.2f}'.format(n_peak, freq)+u'\u212B')
    logger.debug(n_peak)
    logger.debug(fourier[n_peak-1:n_peak+2])
    refined_peak = numpy.argmax(fourier[n_peak-1:n_peak+2]) + n_peak - 1
    noise = fourier[numpy.delete(numpy.arange(fourier.size), numpy.arange(fourier.size)[::refined_peak])]
#    print(noise)
    r = (fourier[refined_peak] - noise.mean()) / (5*noise.std())
    logger.debug('I/5sig value: {:.2f}'.format(r))
    logger.debug('Patterns match!\n') if r>=1.5 else logger.debug('! ! ! ! ! Patterns do not match\n')
    return r.round(2)

def crosscheck2patterns(spots1, spots2, angle_delta, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):
    
    v1, RealCoords1 = find_planes(spots1,
                                  BeamCenter=BeamCenter,
                                  Wavelength=Wavelength,
                                  DetectorDistance=DetectorDistance,
                                  DetectorPixel=DetectorPixel)
    [logger.info('Directions 1st pattern: {0}, confidence {1:.2f}'.format(i[:3], i[3])) for i in v1]
    
    v2, RealCoords2 = find_planes(spots2,
                                  BeamCenter=BeamCenter,
                                  Wavelength=Wavelength,
                                  DetectorDistance=DetectorDistance,
                                  DetectorPixel=DetectorPixel)
    [logger.info('Directions 2nd pattern: {0}, confidence {1:.2f}'.format(i[:3], i[3])) for i in v2]

    matches = []
    conf = []
    for i in v1:
        newi = rotate_vector(i[:3], numpy.array([1.0, 0.0, 0.0]), angle_delta)
        newi = numpy.hstack((newi, i[3]))
        chck = check(RealCoords2, newi)
        matches.append(chck)
        conf.append(newi[3])
    
    for i in v2:
        newi = rotate_vector(i[:3], numpy.array([1.0, 0.0, 0.0]), -angle_delta)
        newi = numpy.hstack((newi, i[3]))
        chck = check(RealCoords1, newi)
        matches.append(chck)
        conf.append(newi[3])
    
    matches = numpy.asarray(matches)
    conf = numpy.asarray(conf)
    return sigmoid(numpy.sum(matches*numpy.exp(conf))/numpy.exp(conf).sum())
#    return numpy.sum(matches*numpy.exp(conf))/numpy.exp(conf).sum()





BeamCenter = (1026.16, 1085.48)
Wavelength = 0.96770
DetectorDistance = 114.60
DetectorPixel = 0.075

angle = -5


#ar1 = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/00001.spot', skiprows=3)
#ar1 = ar1[ar1[:, 3]>numpy.percentile(ar1[:, 3], 80)]
#logger.info('Number of spots 1st file: {}'.format(ar1.shape[0]))
#
#ar_ovlp = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/00901.spot', skiprows=3)
#ar_ovlp = numpy.append(ar_ovlp, numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/01500.spot', skiprows=3), axis=0)
#ar_ovlp = ar_ovlp[ar_ovlp[:, 3]>numpy.percentile(ar_ovlp[:, 3], 0)]
#print(ar_ovlp.shape)
#
##plt.scatter(ar1[:, 1], ar1[:, 2], c='red', marker='+')
#plt.scatter(ar_ovlp[:, 1], ar_ovlp[:, 2], c='blue', marker='x')
#plt.show()

#test = find_planes(ar_ovlp, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)
#print(test)








def projection_reduced_sum(coordinates, normale):
    product = numpy.dot(coordinates, normale)
    z_product = numpy.round(numpy.copy(product))
#    print(z_product)
    delta = numpy.sum(numpy.abs(product - z_product))
    
    return delta

def sphToXYZ(spherical):
    xyz = numpy.zeros(spherical.shape)
    xyz[:, 2] = numpy.cos(spherical[:, 0])/spherical[:, 1]
    xyz[:, 0] = numpy.sin(spherical[:, 0])*numpy.cos(spherical[:, 2])/spherical[:, 1]
    xyz[:, 1] = numpy.sin(spherical[:, 0])*numpy.sin(spherical[:, 2])/spherical[:, 1]
    return xyz

def normale(spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):
#    st = time.time()
    RealCoords = numpy.zeros((numpy.shape(spots)[0], 3))
    
    x = (spots[:, 1] - BeamCenter[0]) * DetectorPixel
    y = (spots[:, 2] - BeamCenter[1]) * DetectorPixel
    divider = Wavelength * numpy.sqrt(x ** 2 + y ** 2 + DetectorDistance ** 2)
    RealCoords[:, 0] = x / divider
    RealCoords[:, 1] = y / divider
    RealCoords[:, 2] = (1/Wavelength) - DetectorDistance/divider
    
    bining = 40
    
    normales = numpy.array(numpy.meshgrid(numpy.linspace(0.0, 3.14, bining),
                                          numpy.logspace(-3, -1, 2*bining, base=10),
                                          numpy.linspace(0.0, 3.14, bining)))
    
    normales = normales.reshape((3, 2*bining**3)).T
    spherical = numpy.copy(normales)
#    plt.plot(normales[:, 1])
#    plt.show()
    
    normales = sphToXYZ(normales)

    NCT = numpy.dot(RealCoords, normales.T)
    NCT_Z = numpy.round(numpy.copy(NCT))
    
    cost = numpy.sum(numpy.abs(NCT - NCT_Z), axis=0)
    
    
    c = 10
    cost -= numpy.convolve(cost, numpy.ones(c), 'same')/c
    thr = numpy.percentile(cost, 0.01)
    
    points = numpy.where(cost<thr)[0]
    
    
    clusters = distance.pdist(spherical[:, ::2][points], metric='euclidean')
    
#    print(distance.squareform(clusters))
#    print(spherical[:, :][points])
    
    plt.imshow(distance.squareform(clusters), cmap='hot')
    plt.colorbar()
    plt.show()
    
    Z = hierarchy.linkage(clusters, method='single', metric='euclidean')
    
    hierarchy.dendrogram(Z, color_threshold=0.5)
    
    F = hierarchy.fcluster(Z, t=0.2, criterion='distance')
    plt.show()
#    print(F)
    
    plt.plot(cost)
    plt.scatter(points, cost[points], marker='o', c='red')
    plt.show()

#    print("{0}Main part finished in {1:.2f} s{2}".format(color.BOLD, (time.time()-st), color.END))
    
    vectors = []
#    for centre in spherical[points]:
    for i in numpy.unique(F):
        centre = spherical[points[F==i]].mean(axis=0)
#        print(centre)
        
#        bining = 41
#        boundary = 0.01
#        mesh = numpy.array(numpy.meshgrid(numpy.linspace(centre[0]-boundary, centre[0]+boundary, bining),
#                                          numpy.linspace(centre[1]-boundary, centre[1]+boundary, bining),
#                                          numpy.linspace(centre[2]-boundary, centre[2]+boundary, bining)))
#        mesh = mesh.reshape((3, bining**3)).T
#        mesh = sphToXYZ(mesh)
#        
#        NCT = numpy.dot(RealCoords, mesh.T)
#        NCT_Z = numpy.round(numpy.copy(NCT))
#    
#        cost = numpy.sum(numpy.abs(NCT - NCT_Z), axis=0)
#        cost -= numpy.convolve(cost, numpy.ones(c), 'same')/c
#        plt.plot(cost)
##        plt.scatter(points, cost[points], marker='o', c='red')
#        plt.show()
        
        maindirection = (centre[0], centre[2]) # theta, phi
        logger.debug('Main direction unrefined: phi={0:.2f}, theta={1:.2f}'.format(maindirection[1], maindirection[0]))
        
        bining = 20
        boundary = 0.05
        
        p = numpy.linspace(maindirection[1]-boundary, maindirection[1]+boundary, bining)
        t = numpy.linspace(maindirection[0]-boundary, maindirection[0]+boundary, bining)

        Z = numpy.zeros((bining, bining))
        for i, j in numpy.ndindex((bining, bining)):
            Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
        plt.imshow(Z, cmap='hot', interpolation='nearest', origin='lower', extent=[p.min(), p.max(), t.min(), t.max()])
        plt.colorbar()
        plt.show()
        refine_peak = find_peaks2D(Z, onepeak=True)
        logger.debug('Refined peak: {}'.format(refine_peak))
#        print(Z[refine_peak[0]])
        
        maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
        
        #iteration
        bining = 10
        boundary = 0.01
        
        p = numpy.linspace(maindirection[1]-boundary, maindirection[1]+boundary, bining)
        t = numpy.linspace(maindirection[0]-boundary, maindirection[0]+boundary, bining)

        Z = numpy.zeros((bining, bining))
        for i, j in numpy.ndindex((bining, bining)):
            Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
        plt.imshow(Z, cmap='hot', interpolation='nearest', origin='lower', extent=[p.min(), p.max(), t.min(), t.max()])
        plt.colorbar()
        plt.show()
        refine_peak = find_peaks2D(Z, onepeak=True)
        logger.debug('Refined peak: {}'.format(refine_peak))
#        print(Z[refine_peak[0]])
        
        maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
        
        bining = 10
        boundary = 0.001
        
        p = numpy.linspace(maindirection[1]-boundary, maindirection[1]+boundary, bining)
        t = numpy.linspace(maindirection[0]-boundary, maindirection[0]+boundary, bining)

        Z = numpy.zeros((bining, bining))
        for i, j in numpy.ndindex((bining, bining)):
            Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
        plt.imshow(Z, cmap='hot', interpolation='nearest', origin='lower', extent=[p.min(), p.max(), t.min(), t.max()])
        plt.colorbar()
        plt.show()
        refine_peak = find_peaks2D(Z, onepeak=True)
        logger.debug('Refined peak: {}'.format(refine_peak))
#        print(Z[refine_peak[0]])

        maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
        logger.debug('Main direction refined: phi={0:.2f}, theta={1:.2f}'.format(maindirection[1], maindirection[0]))

        maindirectionXYZ = backToXYZ(numpy.array([1.0, maindirection[0], maindirection[1]]))
        proj_coords = direction_proj(maindirectionXYZ, RealCoords)
        
        bins = int(0.8*proj_coords.size)
        bins = bins if bins>=50 else 50
        dens = numpy.histogram(proj_coords, bins=bins)[0].astype(float)
        plt.plot(dens)
        plt.show()
        dens -= ndimage.gaussian_filter1d(dens, sigma=0.05*proj_coords.size)
        dens -= numpy.mean(dens)
        fourier = numpy.abs(numpy.fft.rfft(dens))
        plt.plot(fourier)
        plt.show()
        
        peak = int(numpy.argmax(fourier))
        w = numpy.pi*peak/(proj_coords.max()-proj_coords.min())
        noise = fourier[numpy.delete(numpy.arange(fourier.size), numpy.arange(fourier.size)[::peak])]
        score = (fourier[peak] - noise.mean()) / (5*noise.std())
        
        logger.debug('Score for direction: {:.2f}'.format(score))
        if score>=1:
            vectors.append(numpy.hstack((w*maindirectionXYZ, score)))
        else:
            logger.info('Score is too low, aborting this direction: {:.2f}'.format(score))
        
    
    return vectors, RealCoords







#import matplotlib
#matplotlib.use("Agg")

#start = time.time()
#
#
#import time
#spots = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/00001.spot', skiprows=3)
#spots2 = numpy.loadtxt('/home/esrf/melnikov/spyder/test/LYS_dataset/00901.spot', skiprows=3)
##angle_increment = -0.05
#
#
#
##vectors1 = normale(spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)[0]
##vectors2 = find_planes(spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)[0]
#
#angle_delta = -0.25*3.14159
#l = []
#X = crosscheck2patterns(spots, spots2, angle_delta, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)
#
#print("{:.2f}".format(X))
#
#
#plt.close()
#finish = time.time()
#print('Elapsed: {:.2f} s'.format((finish-start)))


















