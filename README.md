# DoubleMesh package


The current package is used to recognise diffraction patterns from different orientations of the same crystal. Typically, if multiple crystals are dispensed onto a sample holder, the problem of optimal centring arises. Especially, when two mesh scans are employed (at orthogonal or any other distinct orientations), one need to make sure that the same crystal is chosen for centring in two scans.

The algorithm within this package is used within a workflow "Two Mesh Scans" used at the ESRF MX beamlines to do multi-crystal data collection, which in turn operates within MXCUBE control software. At the foremost, it requires running of DozorM software beforehand. Then, the DoubleMesh script is run by calling

"doublemesh_lib.analyseDoubleMeshscan(path)"

where "path" is the path to the DozorM directory, where "dozorm2.dat" and "crystal*.spot" files are located from two scans.
