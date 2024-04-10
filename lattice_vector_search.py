# -*- coding: utf-8 -*-
"""
By Igor Melnikov

04/08/2021
"""

import time
import numpy
from scipy import ndimage
from matplotlib import pyplot as plt

__version__ = "3.3"
"""v.3.0 more accurate fourier peak analysis"""
"""v.3.1 fixed logging issues"""
"""v.3.2 DO_PLOT=False within workflows"""
"""v.3.3 some cleaning; added refinement function for plane vector"""


DO_PLOT = True

try:
    from bes.workflow_lib import workflow_logging

    logger = workflow_logging.getLogger()
    DO_PLOT = False
except Exception:
    import logging

    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False


float_formatter = "{:.2f}".format
numpy.set_printoptions(formatter={"float_kind": float_formatter})


class color:
    PURPLE = "\033[1;35;47m"
    GREEN = "\033[1;32;47m"
    RED = "\033[1;31;47m"
    END = "\033[0m"


def sigmoid(x, a=4.39, x0=1.0):
    return 1 / (1 + numpy.exp(-a * (x - x0)))


def sphericalToXYZ(spherical):
    xyz = numpy.zeros(spherical.shape)
    xyz[2] = spherical[0] * numpy.cos(spherical[1])
    xyz[0] = spherical[0] * numpy.sin(spherical[1]) * numpy.cos(spherical[2])
    xyz[1] = spherical[0] * numpy.sin(spherical[1]) * numpy.sin(spherical[2])
    return xyz


def xyzToSpherical(xyz):
    ptsnew = numpy.zeros(xyz.shape)
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptsnew[0] = numpy.sqrt(xy + xyz[2] ** 2)
    ptsnew[1] = numpy.arctan2(numpy.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    # ptsnew[4] = numpy.arctan2(xyz[2], numpy.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[2] = numpy.arctan2(xyz[1], xyz[0])
    return ptsnew


def density(x, sigma, points_coords):
    p = points_coords[:, numpy.newaxis] - x
    return numpy.sum(numpy.exp(-(p**2) / (2 * sigma**2)), axis=0)


def direction_proj(RealCoords, newaxR):
    return numpy.sum(numpy.multiply(newaxR, RealCoords), axis=1)


def decimal(x):
    a1 = x % 1
    a2 = (1 - x) % 1
    return numpy.min(numpy.abs([a1, a2]), axis=0)


def rotate_vector(vector, axis, angle):
    #    logger.debug('Axis: {}'.format(axis))
    axis = axis / numpy.linalg.norm(axis)
    s = numpy.sin(angle)
    c = numpy.cos(angle)
    cp_matr = numpy.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    rotation_matrix = numpy.eye(3) + s * cp_matr + (1 - c) * numpy.dot(cp_matr, cp_matr)

    #    logger.debug('Rotation matrix: {}'.format(rotation_matrix))
    result = numpy.dot(rotation_matrix, vector)
    return result


def find_peaks2D(array2D, onepeak=False):
    if onepeak:
        array2D = ndimage.gaussian_filter(array2D, sigma=2)
#        if DO_PLOT:
#            plt.imshow(array2D, cmap='hot', origin="lower")
#            plt.show()
        return [numpy.unravel_index(numpy.argmax(array2D), array2D.shape)]
    else:
        peak_indices = []
        array2D -= ndimage.gaussian_filter(array2D, sigma=5)
        if DO_PLOT:
            plt.imshow(array2D, cmap='hot')
            plt.imshow(((array2D/(array2D.mean()+3*array2D.std()))>1.0), cmap='hot')
            plt.show()
        zones = ndimage.measurements.label(
            (array2D > array2D.mean() + 3 * array2D.std()), structure=numpy.ones((3, 3))
        )
        #        zones = ndimage.measurements.label((array2D>0.3), structure=numpy.ones((3, 3)))
        for i in range(zones[1]):
            zone = numpy.where(zones[0] == i + 1)
            center = numpy.argmax(array2D[zone])
            center = zone[0][center], zone[1][center]
            peak_indices.append(center)

        return peak_indices


def alignment_score(dr, RealCoords):
    dr = sphericalToXYZ(numpy.array([1, dr[0], dr[1]]))
    newdr_coords = numpy.dot(dr, RealCoords.T)
    histtt = numpy.histogram(newdr_coords, bins=100)[0]
    FT0 = histtt.sum()
    histtt -= ndimage.gaussian_filter1d(histtt, sigma=10)
    ft = numpy.abs(numpy.fft.rfft(histtt))

    return numpy.max(ft) / FT0


def find_planes(spots, BeamCenter, Wavelength, DetectorDistance, DetectorPixel, spots_in3D=False):
    """Returns plane normale vector XYZ with interplane frequency as length and fourier peak height"""
    st = time.time()
    if spots_in3D:
        RealCoords = spots
    else:
        RealCoords = numpy.zeros((numpy.shape(spots)[0], 3))
    
        x = (spots[:, 1] - BeamCenter[0]) * DetectorPixel
        y = (spots[:, 2] - BeamCenter[1]) * DetectorPixel
        divider = Wavelength * numpy.sqrt(x**2 + y**2 + DetectorDistance**2)
        RealCoords[:, 0] = x / divider
        RealCoords[:, 1] = y / divider
        RealCoords[:, 2] = (1 / Wavelength) - DetectorDistance / divider
        #    RealCoords[i, 3] = spots[i, 0]
        #    RealCoords[i, 4] = float(spots[i, 3]) / float(spots[i, 4])

    if len(numpy.atleast_1d(spots)) < 50:
        logger.debug("Not enough spots for proper analysis in some of the crystals")
        return [], RealCoords
    else:
        bining = 50
        phis = numpy.linspace(0, 3.14, bining)
        thetas = numpy.linspace(0, 3.14, bining)

        Z = numpy.zeros((bining, bining))
        for i, j in numpy.ndindex((bining, bining)):
            Z[i, j] = alignment_score((thetas[i], phis[j]), RealCoords)
        if DO_PLOT:
            plt.imshow(
                Z,
                cmap="hot",
                interpolation="nearest",
                origin="lower",
                extent=[phis.min(), phis.max(), thetas.min(), thetas.max()],
            )
            plt.colorbar()
            plt.show()

        logger.debug(
            color.PURPLE
            + "Peak areas: {0}".format(numpy.where(Z > Z.mean() + 10 * Z.std()))
        )

        peaks = find_peaks2D(Z)
        logger.debug("Peaks: {0}".format(peaks))
        bining = 50
        vectors = []
        logger.debug(
            "Coarse search finished in {:.1f} s\n".format((time.time() - st))
            + color.END
        )
        for peak in peaks:
            maindirection = thetas[peak[0]], phis[peak[1]]
            logger.debug(
                "Main direction unrefined: phi={0:.2f}, theta={1:.2f}".format(
                    maindirection[1], maindirection[0]
                )
            )
            bining2 = 11
            boundary = 0.05

            p = numpy.linspace(
                maindirection[1] - boundary, maindirection[1] + boundary, bining2
            )
            t = numpy.linspace(
                maindirection[0] - boundary, maindirection[0] + boundary, bining2
            )

            Z = numpy.zeros((bining2, bining2))
            for i, j in numpy.ndindex((bining2, bining2)):
                Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
            if DO_PLOT:
                plt.imshow(
                    Z,
                    cmap="hot",
                    interpolation="nearest",
                    origin="lower",
                    extent=[p.min(), p.max(), t.min(), t.max()],
                )
                plt.colorbar()
                plt.show()
            refine_peak = find_peaks2D(Z, onepeak=True)
            logger.debug("Refined peak: {}".format(refine_peak))

            maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
            # iteration
            logger.debug(
                "Main direction unrefined: phi={0:.2f}, theta={1:.2f}".format(
                    maindirection[1], maindirection[0]
                )
            )

            boundary = 0.01

            p = numpy.linspace(
                maindirection[1] - boundary, maindirection[1] + boundary, bining2
            )
            t = numpy.linspace(
                maindirection[0] - boundary, maindirection[0] + boundary, bining2
            )

            Z = numpy.zeros((bining2, bining2))
            for i, j in numpy.ndindex((bining2, bining2)):
                Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
            if DO_PLOT:
                plt.imshow(
                    Z,
                    cmap="hot",
                    interpolation="nearest",
                    origin="lower",
                    extent=[p.min(), p.max(), t.min(), t.max()],
                )
                plt.colorbar()
                plt.show()
            refine_peak = find_peaks2D(Z, onepeak=True)
            logger.debug("Refined peak: {}".format(refine_peak))

            maindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
            # iteration fin
            logger.debug(
                "Main direction refined: phi={0:.2f}, theta={1:.2f}".format(
                    maindirection[1], maindirection[0]
                )
            )

            maindirectionXYZ = sphericalToXYZ(
                numpy.array([1.0, maindirection[0], maindirection[1]])
            )
            proj_coords = direction_proj(RealCoords, maindirectionXYZ)

            bins = int(0.8 * proj_coords.size)
            bins = bins if bins >= 50 else 50

            h = numpy.histogram(proj_coords, bins=bins)
            dens = h[0].astype(float)
            corr = ndimage.gaussian_filter1d(dens, sigma=0.1 * bins)
            if DO_PLOT:
                plt.title(
                    "Find: Spot density for direction {0:.1f} {1:.1f} {2:.1f}".format(
                        *list(maindirectionXYZ[:3])
                    )
                    + ", coords size {0}, bins {1}".format(proj_coords.size, bins)
                )
                plt.plot(h[1][:-1] + (h[1][1] - h[1][0]) / 2.0, dens)
                plt.plot(h[1][:-1] + (h[1][1] - h[1][0]) / 2.0, corr)
                plt.xlabel(r"$\AA^{-1}$")
                plt.show()
            dens -= corr
            dens -= numpy.mean(dens)
            fourier = numpy.abs(numpy.fft.rfft(dens))

            peak = int(numpy.argmax(fourier))
            w = numpy.pi * peak / (h[1][-1] - h[1][0])
            #            to_delete = numpy.concatenate([numpy.arange(fourier.size)[::peak+i] for i in range(-(fourier.size//100), (fourier.size//100)+1)])
            #            noise_regions = numpy.delete(numpy.arange(fourier.size), to_delete)
            noise_regions = numpy.arange(fourier.size)[fourier <= numpy.median(fourier)]
            noise_regions = numpy.arange(fourier.size)[fourier <= numpy.median(fourier) + noise_regions.std()]
            noise = fourier[noise_regions]
            frequences = numpy.pi * numpy.arange(fourier.size) / (h[1][-1] - h[1][0])
            
            if DO_PLOT:
                plt.plot(frequences[frequences < 1500], fourier[frequences < 1500])
                plt.plot(
                    frequences[noise_regions][frequences[noise_regions] < 1500],
                    noise[frequences[noise_regions] < 1500],
                    c="red",
                )
                plt.title(
                "Find: Fourier components for direction {0:.1f} {1:.1f} {2:.1f}".format(
                    *list(w * maindirectionXYZ[:3])
                )
                + ", coords size {0}, bins {1}".format(proj_coords.size, bins)
                )   
                plt.xlabel(r"$\AA$")
                plt.show()

            score = (fourier[peak] - noise.mean()) / (5 * noise.std())

            logger.debug("Score (I/5*sig) for direction: {:.2f}".format(score))
            if score >= 0.3:  # >=1:
                vectors.append(numpy.hstack((w * maindirectionXYZ, score)))
            else:
                logger.debug(
                    "Score is too low, aborting this direction: {:.2f}".format(score)
                )
        return vectors, RealCoords


def check(RealCoords, plane_vector):
    freq = numpy.linalg.norm(plane_vector[:3])
    direction = plane_vector[:3] / freq
    proj_coords = direction_proj(RealCoords, direction)
    n_peak = int(freq * (proj_coords.max() - proj_coords.min()) / numpy.pi)
    n_peak = n_peak if n_peak > 2 else 2

    bins = int(max(0.1 * proj_coords.size, 100, 4 * n_peak))
    
    logger.debug("Number of bins: {}".format(bins))

    h = numpy.histogram(proj_coords, bins=bins)
    dens = h[0].astype(float)
    corr = ndimage.gaussian_filter1d(dens, sigma=0.1 * bins)
    if DO_PLOT:
        plt.title(
            "Check: Spot density for direction {0:.1f} {1:.1f} {2:.1f}".format(
                *list(plane_vector[:3])
            )
            + ", coords size {0}, bins {1}".format(proj_coords.size, bins)
        )
        plt.plot(h[1][:-1] + (h[1][1] - h[1][0]) / 2.0, dens)
        plt.plot(h[1][:-1] + (h[1][1] - h[1][0]) / 2.0, corr)
        plt.xlabel(r"$\AA^{-1}$")
        plt.show()
    dens -= corr
    dens -= numpy.mean(dens)
    fourier = numpy.abs(numpy.fft.rfft(dens))

    logger.debug("Search for peak around frequency {:.1f}".format(freq) + "\u212B")
    peaksearch = numpy.array(
        [n_peak + i for i in range(-(bins // 100), (bins // 100) + 1)]
    )
    peaksearch = peaksearch[peaksearch >= 0]
    refined_peak = numpy.argmax(fourier[peaksearch]) + peaksearch[0]
    logger.debug(
        (
            "Values of fourier around peak: {"
            + ":.1f} {".join(numpy.arange(peaksearch.size).astype(str))
            + ":.1f}"
        ).format(*list(fourier[peaksearch]))
    )
    logger.debug(
        "Refined peak position: {:.1f}".format(
            numpy.pi * refined_peak / (h[1][-1] - h[1][0])
        )
        + "\u212B"
    )

    noise_regions = numpy.arange(fourier.size)[fourier <= numpy.median(fourier)]
    noise_regions = numpy.arange(fourier.size)[
        fourier <= numpy.median(fourier) + noise_regions.std()
    ]
    noise = fourier[noise_regions]

    plt.title(
        "Check: Fourier components for direction {0:.1f} {1:.1f} {2:.1f}".format(
            *list(plane_vector[:3])
        )
        + ", coords size {0}, bins {1}, fourier size {2}".format(proj_coords.size, bins, fourier.size)
    )
    frequences = numpy.pi * numpy.arange(fourier.size) / (h[1][-1] - h[1][0])
    if DO_PLOT:
        plt.plot(frequences[frequences < 1500], fourier[frequences < 1500])
        plt.plot(
            frequences[noise_regions][frequences[noise_regions] < 1500],
            noise[frequences[noise_regions] < 1500],
            c="red",
        )
        plt.xlabel(r"$\AA$")
        plt.show()

    r = (fourier[refined_peak] - noise.mean()) / (5 * noise.std())
    logger.debug("I/5sig value: {:.2f}".format(r))
    (
        logger.debug(color.GREEN + "Patterns match!" + color.END)
        if r >= 1.5
        else logger.debug(color.RED + " ! ! ! ! ! Patterns do not match" + color.END)
    )
    return r.round(2)


def crosscheck2patterns(
    spots1, spots2, angle_delta, BeamCenter, Wavelength, DetectorDistance, DetectorPixel, spots_in3D=False
):

    v1, RealCoords1 = find_planes(
        spots1,
        BeamCenter=BeamCenter,
        Wavelength=Wavelength,
        DetectorDistance=DetectorDistance,
        DetectorPixel=DetectorPixel,
        spots_in3D=spots_in3D
    )
    [
        logger.debug(
            "Directions 1st pattern: {0}, confidence {1:.2f}".format(i[:3], i[3])
        )
        for i in v1
    ]

    v2, RealCoords2 = find_planes(
        spots2,
        BeamCenter=BeamCenter,
        Wavelength=Wavelength,
        DetectorDistance=DetectorDistance,
        DetectorPixel=DetectorPixel,
        spots_in3D=spots_in3D
    )
    [
        logger.debug(
            "Directions 2nd pattern: {0}, confidence {1:.2f}".format(i[:3], i[3])
        )
        for i in v2
    ]

    matches = []
    conf = []
    for i in v1:
        newi = rotate_vector(i[:3], numpy.array([1.0, 0.0, 0.0]), -angle_delta)
        newi = numpy.hstack((newi, i[3]))
        chck = check(RealCoords2, newi)
        matches.append(chck)
        conf.append(newi[3])

    for i in v2:
        newi = rotate_vector(i[:3], numpy.array([1.0, 0.0, 0.0]), angle_delta)
        newi = numpy.hstack((newi, i[3]))
        chck = check(RealCoords1, newi)
        matches.append(chck)
        conf.append(newi[3])

    matches = numpy.asarray(matches)
    conf = numpy.asarray(conf)

    return sigmoid(numpy.sum(matches * numpy.exp(conf)) / numpy.exp(conf).sum()), conf.mean()


def refine_lattice_vector(RealCoords, plane_vector):
    original = xyzToSpherical(plane_vector[:3])
    maindirection = original[1:3]
    boundary = 0.01
    bining = 11
    
    p = numpy.linspace(maindirection[1] - boundary, maindirection[1] + boundary, bining)
    t = numpy.linspace(maindirection[0] - boundary, maindirection[0] + boundary, bining)

    Z = numpy.zeros((bining, bining))
    for i, j in numpy.ndindex((bining, bining)):
        Z[i, j] = alignment_score((t[i], p[j]), RealCoords)
    if DO_PLOT:
        plt.imshow(Z, cmap="hot", interpolation="nearest", origin="lower", extent=[p.min(), p.max(), t.min(), t.max()])
        plt.colorbar()
        plt.show()
    
    refine_peak = find_peaks2D(Z, onepeak=True)
    logger.debug("Refined peak: {}".format(refine_peak))

    newmaindirection = t[refine_peak[0][0]], p[refine_peak[0][1]]
    logger.debug("Main direction unrefined: phi={0:.2f}, theta={1:.2f}".format(maindirection[1], maindirection[0]))
    logger.debug("Main direction refined: phi={0:.2f}, theta={1:.2f}".format(newmaindirection[1], newmaindirection[0]))
    
#    delta = numpy.sqrt(((newmaindirection-maindirection)**2).sum())
    delta = newmaindirection-maindirection
    logger.debug("Delta=({0:.3f}, {1:.3f})".format(delta[0], delta[1]))
    
    refinedXYZ = sphericalToXYZ(numpy.array([original[0], newmaindirection[0], newmaindirection[1]]))
    
    logger.debug("Original plane vector: {0:.1f} {1:.1f} {2:.1f}".format(*list(plane_vector[:3])))
    logger.debug("Refined plane vector: {0:.1f} {1:.1f} {2:.1f}".format(*list(refinedXYZ)))
    newratio = check(RealCoords, refinedXYZ)
    
    return delta, newratio
    
