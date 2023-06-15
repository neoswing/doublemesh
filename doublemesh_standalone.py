#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:31:42 2021

@author: melnikov
"""

import time
from doublemesh_lib import *

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
logger.setLevel(logging.DEBUG)

start = time.time()




#try:
#    path = sys.argv[1]
#except:
#    print("Error: Argument 1 is not recognised!\n\
#          \n\
#          Usage: ./doublemesh.py path_to_spots\n\
#          \n\
#          Help: ./doublemesh.py help")
#    quit()
#
#if path=="help":
#    print("Usage: ./doublemesh.py path_to_spots\n\
#          \n\
#          Help: ./doublemesh.py help")
#
#else:
#    start = time.time()
#    analyseDoubleMeshscan(path)



#analyseDoubleMeshscan('./')
#analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20220203/PROCESSED_DATA/Sample-4-1-02/MeshScan_01/Workflow_20220203-135018/DozorM2_mesh-local-user_1_01')

#few spots in this one
#analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20220608/PROCESSED_DATA/test/test-test/MeshScan_02/Workflow_20220608-115431/DozorM2_mesh-test-test_1_01')


#analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20221104/PROCESSED_DATA/Sample-2-1-03/MeshScan_02/Workflow_20221104-131309/DozorM2_mesh-opid231_1_01')



#analyseDoubleMeshscan('/data/id30b/inhouse/opid30b/20230119/PROCESSED_DATA/Sample-8-1-04/MeshScan_01/Workflow_20230119-112331/DozorM2_mesh-Thermolysin_1')

#NEW FORMAT OF INPUT FILE WITH RESOLUTION
#analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20221209/PROCESSED_DATA/RIZK/Sample-2-3-02/MeshScan_06/Workflow_20221209-185832/DozorM2_mesh-opid231_1')

#a, b = analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20230124/PROCESSED_DATA/Sample-1-1-05/MeshScan_03/Workflow_20230124-145916/DozorM2_mesh-local-user_1_01')


#a, b = analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20230426/PROCESSED_DATA/Sasha/TEST2IGOR/Sample-1-1-02/MeshScan_01/Workflow_20230426-141400/DozorM2_mesh-lys_1_01')


#a, b = analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20230503/PROCESSED_DATA/test/test-test/MeshScan_01/Workflow_20230503-103915/DozorM2_mesh-test-test_1_01')


a, b = analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20230606/PROCESSED_DATA/Sample-8-2-01/MeshScan_01/Workflow_20230606-135223/DozorM2_mesh-opid231_1_01')




print(a)
print(b)


logger.info('Elapsed: {:.2f}s'.format(time.time()-start))













