#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:31:42 2021

@author: melnikov
"""


from doublemesh_lib import *

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)






try:
    path = sys.argv[1]
except:
    print("Error: Argument 1 is not recognised!\n\
          \n\
          Usage: ./doublemesh.py path_to_spots\n\
          \n\
          Help: ./doublemesh.py help")
    quit()

if path=="help":
    print("Usage: ./doublemesh.py path_to_spots\n\
          \n\
          Help: ./doublemesh.py help")

else:
    start = time.time()
    analyseDoubleMeshscan(path)



#analyseDoubleMeshscan('./')
#analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20220203/PROCESSED_DATA/Sample-4-1-02/MeshScan_01/Workflow_20220203-135018/DozorM2_mesh-local-user_1_01')

#few spots in this one
#analyseDoubleMeshscan('/data/id23eh1/inhouse/opid231/20220608/PROCESSED_DATA/test/test-test/MeshScan_02/Workflow_20220608-115431/DozorM2_mesh-test-test_1_01')



logger.info('Elapsed: {:.2f}s'.format(time.time()-start))
