# -*- coding: utf-8 -*-
"""
By Igor Melnikov

04/08/2021
"""

import numpy
from scipy import ndimage
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
        array2D = ndimage.gaussian_filter(array2D, sigma=5)
        return [numpy.unravel_index(numpy.argmax(array2D), array2D.shape)]
    else:
        peak_indices = []
        array2D -= ndimage.gaussian_filter(array2D, sigma=1)
        plt.imshow(array2D, cmap='hot')
#        plt.imshow(((array2D/(array2D.mean()+10*array2D.std()))>1.0), cmap='hot')
        plt.show()
        zones = ndimage.measurements.label((array2D>array2D.mean()+10*array2D.std()), structure=numpy.ones((3, 3)))
#        zones = ndimage.measurements.label((array2D>0.3), structure=numpy.ones((3, 3)))
        for i in range(zones[1]):
            zone = numpy.where(zones[0]==i+1)
            center = numpy.argmax(array2D[zone])
            center = zone[0][center], zone[1][center]
            peak_indices.append(center)
        
        return peak_indices
        
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

        bining = 100
        phis = numpy.linspace(0, 3.14, bining)
        thetas = numpy.linspace(0, 3.14, bining)

        Z = numpy.zeros((bining, bining))
        for i, j in numpy.ndindex((bining, bining)):
            Z[i, j] = alignment_score((thetas[i], phis[j]), RealCoords)
        plt.imshow(Z, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
        logger.debug('Peak areas: {0}'.format(numpy.where(Z>Z.mean()+10*Z.std())))
                
        peaks = find_peaks2D(Z)
        logger.debug('Peaks: {0}'.format(peaks))
        bining = 50
        vectors = []
        for peak in peaks:
            maindirection = thetas[peak[0]], phis[peak[1]]
            logger.debug('Main direction unrefined: phi={0:.2f}, theta={1:.2f}'.format(maindirection[1], maindirection[0]))
            
            boundary = 0.05
            
            p = numpy.linspace(maindirection[1]-boundary, maindirection[1]+boundary, 50)
            t = numpy.linspace(maindirection[0]-boundary, maindirection[0]+boundary, 50)

            Z = numpy.zeros((bining, bining))
            for i, j in numpy.ndindex((bining, bining)):
                Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
            plt.imshow(Z, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.show()
            refine_peak = find_peaks2D(Z, onepeak=True)
            logger.debug('Refined peak: {}'.format(refine_peak))
            
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

def check(RealCoords, plane_vector):
    freq = numpy.linalg.norm(plane_vector[:3])
    if freq>300:
        logger.info('Detected large cell parameter ({0:.2f}'.format(freq)+u'\u212B'+
                    '): vector {0} with confidence score {1:.2f}'.format(plane_vector[:3], plane_vector[3]))
        logger.info('No trust to this one')
    direction = plane_vector[:3]/freq
    proj_coords = direction_proj(direction, RealCoords)
    n_peak = int(freq*(proj_coords.max()-proj_coords.min())/numpy.pi)
    
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


ar1 = numpy.loadtxt('test/LYS_dataset/00001.spot', skiprows=3)
ar1 = ar1[ar1[:, 3]>numpy.percentile(ar1[:, 3], 80)]
logger.info('Number of spots 1st file: {}'.format(ar1.shape[0]))

ar_ovlp = numpy.loadtxt('test/LYS_dataset/00901.spot', skiprows=3)
ar_ovlp = numpy.append(ar_ovlp, numpy.loadtxt('test/LYS_dataset/01500.spot', skiprows=3), axis=0)
ar_ovlp = ar_ovlp[ar_ovlp[:, 3]>numpy.percentile(ar_ovlp[:, 3], 0)]
print(ar_ovlp.shape)

#plt.scatter(ar1[:, 1], ar1[:, 2], c='red', marker='+')
plt.scatter(ar_ovlp[:, 1], ar_ovlp[:, 2], c='blue', marker='x')
plt.show()

#test = find_planes(ar_ovlp, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)
#print(test)






import time
spots = numpy.loadtxt('test/LYS_dataset/00001.spot', skiprows=3)



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
    RealCoords = numpy.zeros((numpy.shape(spots)[0], 3))
    
    x = (spots[:, 1] - BeamCenter[0]) * DetectorPixel
    y = (spots[:, 2] - BeamCenter[1]) * DetectorPixel
    divider = Wavelength * numpy.sqrt(x ** 2 + y ** 2 + DetectorDistance ** 2)
    RealCoords[:, 0] = x / divider
    RealCoords[:, 1] = y / divider
    RealCoords[:, 2] = (1/Wavelength) - DetectorDistance/divider
#    print(RealCoords)
    start = time.time()
    
    bining = 50
    
    normales = numpy.array(numpy.meshgrid(numpy.linspace(0.0, 3.14, bining),
                                          numpy.linspace(0.001, 0.1, bining),
                                          numpy.linspace(0.0, 3.14, bining)))
    
    normales = normales.reshape((3, bining**3)).T
    spherical = numpy.copy(normales)
#    plt.plot(normales[:, 1])
#    plt.show()
    
    normales = sphToXYZ(normales)
    print('Elapsed: {:.2f} s'.format((time.time()-start)))
    
    start2 = time.time()
    
    NCT = numpy.dot(RealCoords, normales.T)
    NCT_Z = numpy.round(numpy.copy(NCT))
    
    cost = numpy.sum(numpy.abs(NCT - NCT_Z), axis=0)
    
    finish2 = time.time()
    
    print(cost.shape)
    
    
    thr = numpy.percentile(cost, 0.01)
    points = numpy.where(cost<thr)[0]
    
    
    clusters = distance.pdist(spherical[:, ::2][points], metric='euclidean')
    
    print(distance.squareform(clusters))
    print(spherical[:, ::2][points])
    
    plt.imshow(distance.squareform(clusters), cmap='hot')
    plt.colorbar()
    plt.show()
    
    
    plt.plot(cost)
    plt.scatter(points, cost[points], marker='o', c='red')
    plt.show()

#    for point in points:
#        print(point)
#        direction = normales[point]/numpy.linalg.norm(normales[point])
#        proj_coords = direction_proj(direction, RealCoords)
#        n_peak = int(200*(proj_coords.max()-proj_coords.min())/numpy.pi)
#        
#        bins = int(max(0.04*proj_coords.size, 100, 4*n_peak))
#        logger.debug('Number of bins: {}'.format(bins))
#    
#        dens = numpy.histogram(proj_coords, bins=bins)[0].astype(float)
#        plt.plot(dens)
#        plt.show()
#        dens -= ndimage.gaussian_filter1d(dens, sigma=0.05*proj_coords.size)
#        dens -= numpy.mean(dens)
#        fourier = numpy.abs(numpy.fft.rfft(dens))
#    
#        plt.plot(fourier)
#        plt.show()
    print('Elapsed2: {:.2f} s'.format((finish2-start2)))
    
    
normale(spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel)























#BeamCenter = (1236.51, 1296.86)
#Wavelength = 0.97625
#DetectorDistance = 285.687
#DetectorPixel = 0.172
#
#
#test1 = numpy.loadtxt('../01581.spot', skiprows=3)
#
#test2 = numpy.loadtxt('../00135.spot', skiprows=3)
#print(test1, test2)
#
#test3 = numpy.loadtxt('../01114.spot', skiprows=3)
#
#test4 = numpy.loadtxt('../00167.spot', skiprows=3)
#plt.scatter(test1[:, 1], test1[:, 2], c='blue', marker='x')
#plt.show()
#plt.scatter(test2[:, 1], test2[:, 2], c='red', marker='+')
#plt.show()
#z = crosscheck2patterns(test1, test4, angle_delta=-3.14*0.5,
#                            BeamCenter=BeamCenter,
#                            Wavelength=Wavelength,
#                            DetectorDistance=DetectorDistance,
#                            DetectorPixel=DetectorPixel)
#
#
#print(z)
















#scores = []
#for i in range(50):
#    i = numpy.random.randint(2, 500)
#    name = 'test/LYS_dataset/00'+str(i+2).zfill(3)+'.spot'
#    logger.debug('Next pattern...'+5*'\n')
#    logger.debug('Pattern name: {}'.format(name))
#    ar2 = numpy.loadtxt(name, skiprows=3)
#    ar2 = ar2[ar2[:, 3]>numpy.percentile(ar2[:, 3], 0)]
#
#    logger.info('Number of spots 2nd file: {}'.format(ar2.shape[0]))
#    angle = -0.05*3.14*(i+1)/180.0
#    z = crosscheck2patterns(ar1, ar2, angle_delta=angle,
#                            BeamCenter=BeamCenter,
#                            Wavelength=Wavelength,
#                            DetectorDistance=DetectorDistance,
#                            DetectorPixel=DetectorPixel)
#    logger.info('Pattern match score: {0:.2f}'.format(z))
#    scores.append(z)
#
#scores = numpy.asarray(scores)
##numpy.savetxt('scores_no_match_100spotsx500spots.txt', scores)
#
#logger.info('Score mean: {0:.2f}, StDev: {1:.2f}'.format(scores.mean(), scores.std()))
#
#
#
#
#
#
#
#badscores = []
#for i in range(50):
#    i = numpy.random.randint(2, 500)
#    name = 'test/LYS_dataset/00'+str(i+2).zfill(3)+'.spot'
#    logger.debug('Next pattern...'+5*'\n')
#    logger.debug('Pattern name: {}'.format(name))
#    ar2 = numpy.loadtxt(name, skiprows=3)
#    ar2 = ar2[ar2[:, 3]>numpy.percentile(ar2[:, 3], 0)]
#
#    logger.info('Number of spots 2nd file: {}'.format(ar2.shape[0]))
#    angle = numpy.random.random()*3.14#-0.05*3.14*(i+1)/180.0
#    z = crosscheck2patterns(ar1, ar2, angle_delta=angle,
#                            BeamCenter=BeamCenter,
#                            Wavelength=Wavelength,
#                            DetectorDistance=DetectorDistance,
#                            DetectorPixel=DetectorPixel)
#    logger.info('Pattern match score: {0:.2f}'.format(z))
#    badscores.append(z)
#
#badscores = numpy.asarray(badscores)
##numpy.savetxt('scores_no_match_100spotsx500spots.txt', scores)
#
#logger.info('Score mean: {0:.2f}, StDev: {1:.2f}'.format(badscores.mean(), badscores.std()))
#
#
#
#
#plt.plot(scores)
#plt.plot(badscores)
#plt.show()








