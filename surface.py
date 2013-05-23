from __future__ import print_function
import numpy as np
import logging
import time
from itertools import count, izip
from numpy.random import random, seed
from entity import Entity
try:
    from _refine_fast import _refine_point_list
    REFINE_FAST = True
except ImportError:
    print("Warning, fast refining not available")
    REFINE_FAST = False
    

def refine_voxel(vertices, points, thres):
    """
    Inter-voxel refinement with trilinear interpolation.
    """
    vertices = vertices - thres
    c00 = vertices[0]*(1-points[0, :]) + vertices[4]*points[0, :]
    c01 = vertices[1]*(1-points[0, :]) + vertices[5]*points[0, :]
    c10 = vertices[2]*(1-points[0, :]) + vertices[6]*points[0, :]
    c11 = vertices[3]*(1-points[0, :]) + vertices[7]*points[0, :]

    c0 = c00*(1-points[1, :]) + c01*points[1, :]
    c1 = c10*(1-points[1, :]) + c11*points[1, :]

    residue = (c0*(1-points[2, :]) + c1*points[2, :])**2

    return points[:, residue.argmin()], residue.min()


class Surface(Entity):
    def __init__(self, voxels, thres):
        super(Surface, self).__init__(voxels)
        self.thres = thres
        self.nvox = len(self.voxels[0])

    def point_list(self, field):
        """
        Returns the surface as a point list for distance computation.
        """
        xgrid = 0.5*(field.xr[1:]+field.xr[:-1])
        ygrid = 0.5*(field.yr[1:]+field.yr[:-1])
        zgrid = 0.5*(field.zr[1:]+field.zr[:-1])

        pnts = np.empty((len(self.voxels[0]), 3), dtype=np.double)
        for i, j, k, cursor in izip(self.voxels[0], self.voxels[1], self.voxels[2], count()):
            pnts[cursor, 0] = xgrid[i]
            pnts[cursor, 1] = ygrid[j]
            pnts[cursor, 2] = zgrid[k]

        return pnts

    def refined_point_list(self, field, NGUESS=20, fast=True):
        """
        Returns the surface as a point list, with refinement, for distance
        computation.
        """
        pnts = np.empty((self.nvox, 3), dtype=np.double)
        now = time.clock()
        seed(int(now*3)) #Seed the random number generator
        if REFINE_FAST and fast:
            logging.info("Using Cython implementation of surface refining")
            retval = _refine_point_list(self.nvox,
                                        self.voxels[0],
                                        self.voxels[1],
                                        self.voxels[2],
                                        field.xr,
                                        field.yr,
                                        field.zr,
                                        field.data,
                                        pnts,
                                        random((3, NGUESS)),
                                        self.thres,
                                        NGUESS)
        else:
            vertices = np.zeros((8, ), dtype=np.float32)
            res = np.empty((self.nvox, ), dtype=np.double)

            for i, j, k, cursor in izip(self.voxels[0], self.voxels[1], self.voxels[2], count()):
                dx = field.xr[i+1]-field.xr[i]
                dz = field.zr[k+1]-field.zr[k]
                dy = field.yr[j+1]-field.yr[j]
                
                vertices[:] = field.data[i:i+2, j:j+2, k:k+2].flatten()
                
                guess = random((3, NGUESS))
                opt, res[cursor] = refine_voxel(vertices, guess, self.thres)
                pnts[cursor, 0] = opt[0]*dx + field.xr[i]
                pnts[cursor, 1] = opt[1]*dy + field.yr[j]
                pnts[cursor, 2] = opt[2]*dz + field.zr[k]

        logging.info('Refining took {} s.'.format(time.clock()-now))
        return pnts

    def oversampled_point_list(self, field, NGUESS=20, PASS=5, fast=True):
        pnts = np.empty((PASS*self.nvox, 3), dtype=np.double)
        print("Warning, oversampling the surface {} times".format(PASS))
        for i in range(PASS):
            print("Pass {}".format(i))
            pnts[i*self.nvox:(i+1)*self.nvox, :] = self.refined_point_list(
                field, NGUESS, fast)

        return pnts
