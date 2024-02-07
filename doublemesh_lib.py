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
import numpy
import billiard
import warnings
import matplotlib

try:
    import lattice_vector_search
except ModuleNotFoundError:
    from bes.workflow_lib import lattice_vector_search

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # Noqa E402


__version__ = "1.8.1"

"""added output as first position in case no matches have been found"""
"""added version notice"""

warnings.filterwarnings("ignore")


def extractDozorMetadata(datfilename):
    try:
        from bes.workflow_lib import workflow_logging

        logger = workflow_logging.getLogger()
    except Exception:
        import logging

        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)

    with open(datfilename, "r") as metadata:
        lines = metadata.readlines()
    metadata.close()

    orgx, orgy, Wavelength, DetectorDistance, DetectorPixel, Phi1, Phi2 = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    for line in lines:
        if re.match("orgx", line):
            orgx = float(line.split(" ")[-1])
        if re.match("orgy", line):
            orgy = float(line.split(" ")[-1])
        if re.match("X-ray_wavelength", line):
            Wavelength = float(line.split(" ")[-1])
        if re.match("detector_distance", line):
            DetectorDistance = float(line.split(" ")[-1])
        if re.match("pixel", line):
            DetectorPixel = float(line.split(" ")[-1])
        if re.match("phi1", line):
            Phi1 = float(line.split(" ")[-1])
        if re.match("phi2", line):
            Phi2 = float(line.split(" ")[-1])

    if (
        isinstance(orgx, float)
        and isinstance(orgy, float)
        and isinstance(Wavelength, float)
        and isinstance(DetectorDistance, float)
        and isinstance(DetectorPixel, float)
        and isinstance(Phi1, float)
        and isinstance(Phi2, float)
    ):
        angle_delta = Phi2 - Phi1
        return (orgx, orgy), Wavelength, DetectorDistance, DetectorPixel, angle_delta
    else:
        logger.error("Problem in reading dat file")
        return


def findPlanes_MP(queue, BeamCenter, Wavelength, DetectorDistance, DetectorPixel):
    global Buffer
    while True:
        spots, crystal_n = queue.get()
        if not isinstance(spots, numpy.ndarray):
            break

        v, RealCoords = lattice_vector_search.find_planes(
            spots,
            BeamCenter=BeamCenter,
            Wavelength=Wavelength,
            DetectorDistance=DetectorDistance,
            DetectorPixel=DetectorPixel,
        )
        Buffer[crystal_n] = v, RealCoords


def analyseDoubleMeshscan(path):
    try:
        from bes.workflow_lib import workflow_logging

        logger = workflow_logging.getLogger()
    except Exception:
        import logging

        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)

    start = time.time()
    logger.info("doublemesh_lib, version: {}".format(__version__))
    logger.info(
        "lattice_vector_search, version: {}".format(lattice_vector_search.__version__)
    )
    global Buffer
    initialCWD = os.getcwd()
    os.chdir(path)

    crystals1 = glob.glob("crystal_1_*.spot")
    crystals1.sort()
    crystals2 = glob.glob("crystal_2_*.spot")
    crystals2.sort()

    crystals_mesh1 = [
        numpy.loadtxt(name, skiprows=1, dtype=float) for name in crystals1
    ]
    crystals_mesh2 = [
        numpy.loadtxt(name, skiprows=1, dtype=float) for name in crystals2
    ]

    #    crosscheck_matrix = numpy.zeros((len(crystals_mesh1), len(crystals_mesh2)))

    BeamCenter, Wavelength, DetectorDistance, DetectorPixel, angle_delta = (
        extractDozorMetadata(glob.glob("dozorm2.dat")[0])
    )

    logger.debug(
        "Experiment metadata: BeamCenter {0} {1}, Wavelength {2}, DtoX {3}, Pixelsize {4}, Omega difference {5}".format(
            BeamCenter[0],
            BeamCenter[1],
            Wavelength,
            DetectorDistance,
            DetectorPixel,
            angle_delta,
        )
    )

    angle_delta = angle_delta * 3.14 / 180.0

    input_table = numpy.loadtxt("coordinat_list.dat", skiprows=0, ndmin=2)

    #    potentialMatches = numpy.loadtxt('dozorm_pair.dat')
    potentialMatches = input_table[:, [0, 1, 2]]
    potentialMatches = numpy.hstack(
        (potentialMatches, numpy.zeros((potentialMatches.shape[0], 2)))
    )

    manager = billiard.Manager()
    Buffer = manager.dict()
    nCPU = billiard.cpu_count()
    logger.info("CPU count: {}".format(nCPU))
    queue = billiard.Queue()

    i = 0
    for spots in crystals_mesh1:
        spots = numpy.hstack((numpy.zeros((spots.shape[0], 1)), spots))
        #        spots = spots[spots[:, -1]>spots[:, -1].max()/2.0]
        #        print(spots.shape)
        queue.put((spots, i))
        i += 1
    for item in range(nCPU):
        queue.put((None, None))

    workers = []
    for item in range(nCPU):
        worker = billiard.Process(
            target=findPlanes_MP,
            args=(
                queue,
                BeamCenter,
                Wavelength,
                DetectorDistance,
                DetectorPixel,
            ),
        )
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()

    Buffer0 = Buffer
    Buffer = manager.dict()

    i = 0
    for spots in crystals_mesh2:
        spots = numpy.hstack((numpy.zeros((spots.shape[0], 1)), spots))
        #        spots = spots[spots[:, -1]>spots[:, -1].max()/2.0]
        #        print(spots.shape)
        queue.put((spots, i))
        i += 1
    for item in range(nCPU):
        queue.put((None, None))

    workers = []
    for item in range(nCPU):
        worker = billiard.Process(
            target=findPlanes_MP,
            args=(
                queue,
                BeamCenter,
                Wavelength,
                DetectorDistance,
                DetectorPixel,
            ),
        )
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()

    logger.debug("\n\n\n--------Finished with identifying plane vectors--------\n\n\n")
    logger.debug("Identified vectors for \033[1;32;47mmesh1\033[0m:\n")
    for i in range(len(crystals1)):
        logger.debug("\033[1;32;47mCrystal {0:03d}:\033[0m".format(i + 1))
        for vector in Buffer0[i][0]:
            logger.debug(
                "{0:03.1f} {1:03.1f} {2:03.1f} score {3:.1f}".format(*list(vector))
            )

    logger.debug("Identified vectors for \033[1;34;47mmesh2\033[0m:\n")
    for i in range(len(crystals2)):
        logger.debug("\033[1;34;47mCrystal {0:03d}:\033[0m".format(i + 1))
        for vector in Buffer[i][0]:
            logger.debug(
                "{0:03.1f} {1:03.1f} {2:03.1f} score {3:.1f}".format(*list(vector))
            )

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
            newvi = lattice_vector_search.rotate_vector(
                vi[:3], numpy.array([1.0, 0.0, 0.0]), -angle_delta
            )
            newvi = numpy.hstack((newvi, vi[3]))
            chck = lattice_vector_search.check(RealCoordsj, newvi)
            matches.append(chck)
            conf.append(newvi[3])
        for vj in vectorsj:
            newvj = lattice_vector_search.rotate_vector(
                vj[:3], numpy.array([1.0, 0.0, 0.0]), angle_delta
            )
            newvj = numpy.hstack((newvj, vj[3]))
            chck = lattice_vector_search.check(RealCoordsi, newvj)
            matches.append(chck)
            conf.append(newvj[3])

        matches = numpy.asarray(matches)
        conf = numpy.asarray(conf)
        #        conf = lattice_vector_search.sigmoid1(conf)
        #        line[3] = numpy.multiply(matches, conf).mean()
        line[3] = lattice_vector_search.sigmoid(
            numpy.sum(matches * numpy.exp(conf)) / numpy.exp(conf).sum()
        )
        line[4] = conf.mean()

    potentialMatches = numpy.hstack(
        (
            potentialMatches,
            (potentialMatches[:, 3] > 0.5).reshape(potentialMatches.shape[0], 1),
        )
    )

    potentialMatches = numpy.hstack(
        (potentialMatches, numpy.zeros((potentialMatches.shape[0], 1)))
    )

    treated = numpy.array([], dtype="int")
    for cycle in range(1000):
        verified = numpy.delete(potentialMatches, treated, axis=0)

        verified = verified[verified[:, 5].astype(bool)]
        if verified.size > 0:
            thrsh = numpy.percentile(verified[:, 3], 99)
            candidates = verified[verified[:, 3] >= thrsh]
            x = verified[numpy.argmax(candidates[:, 4]), 0].astype(int) - 1

        else:
            break
        potentialMatches[x, 6] = 1

        cr1 = potentialMatches[x, 1]
        cr2 = potentialMatches[x, 2]

        excl = numpy.unique(
            numpy.append(
                numpy.where(potentialMatches[:, 1] == cr1)[0],
                numpy.where(potentialMatches[:, 2] == cr2)[0],
            )
        )

        treated = numpy.append(treated, excl)

    # adding aperture
    potentialMatches = numpy.hstack((potentialMatches, input_table[:, [4, 5, 6, 7, 8]]))
    #    commands = ['mv(sampx, {0:.4f}, sampy, {1:.4f}, phiy, {2:.4f})'.format(item[8], item[9], item[10]) for item in potentialMatches]
    logger.info("Calculation finished!")
    # potentialMatches: Case Xtal1 Xtal2 MatchScore Confidence Y/N Collect? Resolution BeamSize SampX SampY PhiY

    # If no certain positions have been identified:
    if not numpy.all(potentialMatches[:, 6]):
        logger.info(
            "\033[1;31;47mNo certain matches has been found. Trying to collect at the most probable position.\033[0m"
        )
        potentialMatches[0, 6] = True

    logger.info(
        "Case# | Xtal1 | Xtal2 |  Score  | Confidence | Y/N | Collect? | Resolution | Beam size |        Center command  "
    )
    for item in potentialMatches:
        logger.info(
            "{0:3.0f}   | {1:3.0f}   | {2:3.0f}   |  {3:4.2f}   |  {4:>7s}   |  {5}  |    {6}     |    {7:.2f}    |     {8:3.0f}   |  {9} ".format(
                item[0],
                item[1],
                item[2],
                item[3],
                format(item[4], "4.2f"),
                "Y" if item[5].astype(bool) else "N",
                "Y" if item[6].astype(bool) else " ",
                item[7],
                item[8],
                "mv(sampx, {0:.4f}, sampy, {1:.4f}, phiy, {2:.4f})".format(
                    item[9], item[10], item[11]
                ),
            )
        )

    #    numpy.savetxt('dozorm_pair_final.dat', potentialMatches, fmt='%d %d %d %.2f %3.2f %d %d %1.4f %3d %1.4f %1.4f %1.4f')
    plt.close()

    os.chdir(initialCWD)

    #    collectPosition columns: 0_resolution 1_beam_size 2_sampx 3_sampy 4_phiy
    collectPositions = potentialMatches[potentialMatches[:, 6].astype(bool)][:, 7:]
    logger.info("Elapsed: {:.2f}s".format(time.time() - start))
    return collectPositions, potentialMatches


# matplotlib.use('Agg')

# analyseDoubleMeshscan('./')
# analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20220203/PROCESSED_DATA/Sample-4-1-02/MeshScan_01/Workflow_20220203-135018/DozorM2_mesh-local-user_1_01')
# analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20220608/PROCESSED_DATA/test/test-test/MeshScan_02/Workflow_20220608-115431/DozorM2_mesh-test-test_1_01')
# analyseDoubleMeshscan(
#     "/data/id23eh1/inhouse/opid231/20240117/PROCESSED_DATA/GORD/OLPVR/olpvr/olpvr-x/run_01_MeshScan/Workflow_20240117-164009/DozorM2_two_meshes"
# )


# potentialMatches = numpy.loadtxt('/data/id23eh1/inhouse/opid231/20220203/PROCESSED_DATA/Sample-4-1-02/MeshScan_01/Workflow_20220203-135018/DozorM2_mesh-local-user_1_01/test.dat',
#                                 skiprows=5)[:, [0,2,3]]
# print(potentialMatches[potentialMatches[:, 2]>2])
# x = numpy.unique(potentialMatches[:, 1], return_counts=True)
#
# print(x[0][x[1]>1])


# 30092023
# analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20230926/PROCESSED_DATA/br/br-br4/run_01_MeshScan/Workflow_20230926-195834/DozorM2_mesh-br-br4_1_2_01')


# 07122023
# analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20231207/PROCESSED_DATA/Sample-2-2-02/run_01_MeshScan/Workflow_20231207-143206/DozorM2_mesh-opid231_1_2_01')


# analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20231212/PROCESSED_DATA/Sample-3-2-04/run_04_MeshScan/Workflow_20231212-154107/DozorM2_two_meshes')
