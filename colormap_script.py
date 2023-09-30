#!/usr/bin/env python3
'''
Created on Sep 18, 2023

@author: melnikov
'''

import re
import numpy
import glob
from matplotlib import colors, pyplot as plt
from matplotlib.patches import Circle as C
from scipy import ndimage, signal







def parser(filename):
    l = []
    with open(filename, 'r') as f:
        l = f.readlines()
        f.close()
    
#    print(re.split(r'\s+', l[1])[1:3])
    col, row = [int(x) for x in re.split(r'\s+', l[1])[1:3]]
    
#    Dtable
    Dtable = numpy.zeros((row, col))
    i = -4
    for line in l:
        if re.search('Map of Scores', line):
            i = -3
        
        if i>0:
#            print(numpy.array(re.findall('.{6}', line[5:])).astype(float).size)
            Dtable[i, :] = numpy.array(re.findall('.{6}', line[5:])).astype(float)

        if i>=row-1:
            break

        if i>=-3:
            i += 1
    
#    Ztable
    Ztable = numpy.zeros((row, col))
    i = -4
    for line in l:
        if re.search('Map of Crystals', line):
            i = -3
        
        if i>0:
#            print(numpy.array(re.findall('.{4}', line[5:])).astype(int).size)
            Ztable[i, :] = numpy.array(re.findall('.{4}', line[5:])).astype(int)

        if i>=row-1:
            break

        if i>=-3:
            i += 1

    
#    plt.imshow(Dtable)
#    plt.show()

    return Dtable, Ztable


#parser('/data/id23eh1/inhouse/opid231/20230801/PROCESSED_DATA/DOZORM2_TESTMB/1/MeshScan_01/Workflow_20230801-114054/DozorM2_mesh-mb_1_01/dozorm_001.map')



def ConstructColorlist(array):
    basecolors = [u'#00CA02', u'#FF0101', u'#F5A26F', u'#668DE5', u'#E224DE', u'#04FEFD', u'#FEFE00', u'#0004AF', u'#B5FF06']

    N = int(numpy.max(array))
    AdjacentArray = numpy.identity(N)

    t = numpy.ones((3, 3), dtype='int32')
    for j in range(1, N + 1):
        cut = ndimage.measurements.label(array == j)
        c = signal.convolve2d(cut[0], t, mode='same')
        adjacentvalues = numpy.unique(array[numpy.where(c != 0)]).astype('int')

        for i in adjacentvalues:
            if i == -1 or i == -2:
                pass
            else:
                AdjacentArray[j - 1, i - 1] = 1
    
#    print(AdjacentArray)
    
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Adjacent_NULL = numpy.tril(AdjacentArray, k= -1)
    AdjacentArray = Adjacent_NULL

    ColorVector = numpy.ones(N)
    for i in range(N):
        BannedColors = numpy.unique(AdjacentArray[i, :])[1:]
        for item in BannedColors:
            t.remove(item)
        ColorVector[i] = t[0]
        AdjacentArray = ColorVector * Adjacent_NULL
        t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ColorVector = ColorVector.astype(int)

#    random.shuffle(basecolors)
    Colors = ColorVector.astype(str)
    for i in range(N):
        Colors[i] = basecolors[ColorVector[i] - 1]

    clrs = ['grey', 'black', 'black']
    clrs.extend(Colors.tolist())
    
#    print(clrs)
    
    return clrs





def MainPlot(filename):
    
    fig, ax = plt.subplots()
    
    Dtable, Ztable = parser(filename)
    row, col = Dtable.shape
    
    Zcopy = numpy.copy(Ztable)
    Ztable = numpy.abs(Ztable)
    Ztable[Ztable==999] = -2
    
#    print(numpy.sort(numpy.unique(Ztable[Ztable>0])))
    
    clrs = ConstructColorlist(Ztable)
#    print(clrs)

#    logger.debug('Checkpoint for colormap function:', (time.time()-start_time))
    Ncolors = len(clrs)
    cmap = colors.ListedColormap([colors.to_rgba(str(i)) for i in clrs])

    cmap2 = colors.LinearSegmentedColormap.from_list('my_cmap2', [(0, 0, 0), (0, 0, 0.001)], 256)
    cmap2._init()
    alphas = numpy.linspace(-0.01, -1, cmap2.N + 3)
    cmap2._lut[:, -1] = alphas

    bounds = numpy.linspace(-2, Ncolors - 2, Ncolors + 1)
    norm = colors.BoundaryNorm(bounds, ncolors=Ncolors)
    
    CrystImage = plt.imshow(Ztable, cmap=cmap, norm=norm, interpolation='nearest', origin='upper', \
                            extent=[0.5, (col + 0.5), (row + 0.5), 0.5])
#    OpacityImage = plt.imshow(Dtable, cmap=cmap2, interpolation='nearest', origin='upper', vmin=0.0, \
#                              vmax=numpy.percentile(Dtable, 99), extent=[0.5, (col + 0.5), (row + 0.5), 0.5])
    
    
    opacity = 1+cmap2(Dtable)[:, :, 3]
    rgb = cmap(norm(Ztable))
    
    hexArray = numpy.empty(numpy.shape(opacity), dtype='U7')
    for i, v in numpy.ndenumerate(opacity):
        hexArray[i] = colors.rgb2hex(v*rgb[i][:3])
        
    m = int(round(numpy.log10(numpy.max(Dtable))-2, 0))
    M = numpy.max(Dtable)
    for (j, i) in numpy.ndindex((row, col)):
        if Ztable[j, i] > 0:
            if (j, i + 1) in numpy.ndindex((row, col)):
                if Ztable[j, i + 1] != Ztable[j, i]:
                    line = plt.Line2D((i + 1.5, i + 1.5), (j + 0.5, j + 1.5), lw=2, color='white')
                    plt.gca().add_line(line)
            if (j, i - 1) in numpy.ndindex((row, col)):
                if Ztable[j, i - 1] != Ztable[j, i]:
                    line = plt.Line2D((i + 0.5, i + 0.5), (j + 0.5, j + 1.5), lw=2, color='white')
                    plt.gca().add_line(line)
            if (j + 1, i) in numpy.ndindex((row, col)):
                if Ztable[j + 1, i] != Ztable[j, i]:
                    line = plt.Line2D((i + 0.5, i + 1.5), (j + 1.5, j + 1.5), lw=2, color='white')
                    plt.gca().add_line(line)
            if (j - 1, i) in numpy.ndindex((row, col)):
                if Ztable[j - 1, i] != Ztable[j, i]:
                    line = plt.Line2D((i + 0.5, i + 1.5), (j + 0.5, j + 0.5), lw=2, color='white')
                    plt.gca().add_line(line)

            plt.text(i+1, j+1, Zcopy[j, i].astype(int), c='black', ha='center', va='center', size=3)
            ax.add_patch(C((i+1, j+1), Dtable[j, i]/(3*M), color='white'))
        elif Ztable[j, i]==-2:
#            plt.text(i+1, j+1, ('{:.'+str(m)+'f}').format(Dtable[j, i]), c='white', ha='center', va='center', size=5)
            ax.add_patch(C((i+1, j+1), Dtable[j, i]/(3*M), color='white'))
    
    plt.xticks(numpy.arange(1, Ztable.shape[1]+1, 2), rotation=45)
    plt.yticks(numpy.arange(1, Ztable.shape[0]+1, 2))
    
    plt.text(col+1, 2, 'Dozor\nscore', c='black', ha='left', va='center', size=10)
    ax.add_patch(C((col+1.5, 4), 0.333, color='black', clip_on=False))
    plt.text(col+2, 4, ('{:.'+str(m)+'f}').format(M), c='black', ha='left', va='center', size=10)
    
#    plt.colorbar()
    plt.savefig('CrystalMap'+filename[-7:-4]+'.png', dpi=150)
#    plt.show()
    plt.close()

#MainPlot('/data/id23eh1/inhouse/opid231/20230801/PROCESSED_DATA/DOZORM2_TESTMB/1/MeshScan_01/Workflow_20230801-114054/DozorM2_mesh-mb_1_01/dozorm_001.map')


for item in glob.glob('dozor*.map'):
    MainPlot(item)

