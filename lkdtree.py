import numpy as np
import ctypes as C
import unittest

_lkdtree = np.ctypeslib.load_library('libkdtree','.')
_lkdtree.distances.restype = C.c_int
_lkdtree.distances.argtypes = [C.POINTER(C.c_double),
                               C.c_int,
                               C.POINTER(C.c_double),
                               C.c_int,
                               C.POINTER(C.c_double)]


def distances(voxels, focus):
    NPOINTS = voxels.shape[0]
    NDIST = focus.shape[0]
    #print "{} distance computations over {} voxels".format(NDIST,NPOINTS)
    dist = np.empty((NDIST,), dtype=np.double)

    retval = _lkdtree.distances(
        voxels.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(NPOINTS),
        focus.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(NDIST),
        dist.ctypes.data_as(C.POINTER(C.c_double)))

    del retval

    return np.sqrt(dist)


class TestDistance(unittest.TestCase):
    def test_easy_distance(self):
        """
        Compute three distance computations for two voxels
        """
        point = np.array([[0., 0., 0.],
                          [0., 0., 1.]], dtype=np.double)
        target = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.],
                           [0., 0., 1.5]], dtype=np.double)
        dist = distances(point, target)
        self.assertTrue((dist == np.array([1., 1., 0., 0.5], dtype=np.double)).all())
