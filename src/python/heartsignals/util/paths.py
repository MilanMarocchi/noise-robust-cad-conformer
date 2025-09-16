"""
    paths.py

    returns the path to the matlab code
    AUthor : Milan Marocchi
"""

import os

# The paths of this project
UTIL_PATH = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.dirname(UTIL_PATH)
PYTHON_PATH = os.path.dirname(LIB_PATH)
PROCESSING_PATH = os.path.join(LIB_PATH, "processing")
ROOT = os.path.dirname(PYTHON_PATH)
MATLAB_PATH = os.path.abspath(os.path.join(ROOT, "matlab"))
PROJECT_PATH = os.path.dirname(ROOT)

# expected path to ephnogram and mit
EPHNOGRAM = os.path.join(PROJECT_PATH, 'data', 'ephnogram', 'WFDB')
MIT = os.path.join(PROJECT_PATH, 'data', 'mit-bih-noise-stress-test-database-1.0.0')
