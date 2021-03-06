from __future__ import print_function
import numpy as np
from scipy.ndimage.measurements import label, find_objects
from entity import Entity
from surface import Surface
from numpy.random import randint

#imports for testing
import unittest
import logging
from itertools import product


class Field(object):
    def __init__(self, data, xr, yr, zr):
        self.data = data
        self.xr = xr
        self.yr = yr
        self.zr = zr

    @property
    def NX(self):
        return self.data.shape[0]

    @property
    def NY(self):
        return self.data.shape[1]

    @property
    def NZ(self):
        return self.data.shape[2]

    def label_gt_largest_mask(self, thres):
        """
        Extracts the largest connected entity from the flow. Returns a
        mask array.
        """
        #First, get the largest complimentary structure.
        labeled, nsurf = label(self.data < thres)
        volumes = np.empty((nsurf,), dtype=np.int)
        found = False
        
        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            vol = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])
            if vol > labeled.shape[0]*labeled.shape[2]:
                compvol = (labeled == objnum+1)
                logging.info(
                    'Found volume {} big. Stopping'.format(
                        np.count_nonzero(compvol)))
                found = True
                break

            volumes[objnum] = vol

        if not found:
            compvol = (labeled == volumes.argmax()+1)
 
        #Now, label the not(the large complimentary)
        labeled, nsurf = label(np.bitwise_not(compvol))
        volumes = np.empty((nsurf,), dtype=np.int)

        #Return only the largest.        
        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            vol = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])
            if vol > labeled.shape[0]*labeled.shape[2]:
                compvol = (labeled == objnum+1)
                logging.info(
                    'Found volume {} big. Stopping'.format(
                        np.count_nonzero(compvol)))
                return compvol

            volumes[objnum] = vol

        compvol = (labeled == volumes.argmax()+1)
        logging.info(
            'Found volume {} for {} total'.format(
                np.count_nonzero(compvol),np.prod(labeled.shape))
            )
        return compvol 

    def label_gt_largest(self, thres):
        """
        Extracts the largest connected entity from the flow. Higher
        than the threshold. Fast.
        """
        #First, get the largest complimentary structure.
        labeled, nsurf = label(self.data < thres)
        volumes = np.empty((nsurf,), dtype=np.int)

        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])

        #Now, label the not(the large complimentary)
        labeled, nsurf = label(np.bitwise_not(labeled == volumes.argmax()+1))
        volumes = np.empty((nsurf,), dtype=np.int)
        
        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])
        
        #Return only the largest.
        return Entity(np.where(labeled == volumes.argmax()+1))


    def label_lt_largest(self, thres):
        """
        Extracts all the connected entities from the flow. Lower than
        the threshold
        """
        #First, get the largest complimentary structure.
        labeled, nsurf = label(self.data > thres)
        volumes = np.empty((nsurf,), dtype=np.int)

        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])

        #Now, label the not(the large complimentary)
        labeled, nsurf = label(np.bitwise_not(labeled == volumes.argmax()+1))
        volumes = np.empty((nsurf,), dtype=np.int)
        
        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])
        
        #Return only the largest.
        return Entity(np.where(labeled == volumes.argmax()+1))


    def label_surfaces(self,thres):
        """
        Return an array with all the surfaces labelled
        """
        ens = self.data
        mask = ens > thres
        fill = np.empty((ens.shape[0]-1, ens.shape[1]-1, ens.shape[2]-1),
                        dtype=np.bool)
        void = np.empty((ens.shape[0]-1, ens.shape[1]-1, ens.shape[2]-1),
                        dtype=np.bool)

        fill[:, :] = np.bitwise_and(mask[:-1, :-1, :-1], mask[:-1, :-1, 1:])
        fill[:, :] = np.bitwise_and(fill[:, :], mask[:-1, 1:, :-1])
        fill[:, :] = np.bitwise_and(fill[:, :], mask[:-1, 1:, 1:])
        fill[:, :] = np.bitwise_and(fill[:, :], mask[1:, :-1, :-1])
        fill[:, :] = np.bitwise_and(fill[:, :], mask[1:, :-1, 1:])
        fill[:, :] = np.bitwise_and(fill[:, :], mask[1:, 1:, :-1])
        fill[:, :] = np.bitwise_and(fill[:, :], mask[1:, 1:, 1:])

        mask = ens[:, :, :] < thres
        void[:, :] = np.bitwise_and(mask[:-1, :-1, :-1], mask[:-1, :-1, 1:])
        void[:, :] = np.bitwise_and(void[:, :], mask[:-1, 1:, :-1])
        void[:, :] = np.bitwise_and(void[:, :], mask[:-1, 1:, 1:])
        void[:, :] = np.bitwise_and(void[:, :], mask[1:, :-1, :-1])
        void[:, :] = np.bitwise_and(void[:, :], mask[1:, :-1, 1:])
        void[:, :] = np.bitwise_and(void[:, :], mask[1:, 1:, :-1])
        void[:, :] = np.bitwise_and(void[:, :], mask[1:, 1:, 1:])

        labeled, nsurf = label(np.bitwise_not(np.bitwise_or(fill, void)))
        return labeled,nsurf

    def extract_complete_surface(self,thres):
        labeled, nsurf = self.label_surfaces(thres)
        return Surface(np.where(labeled >= 1), thres)

    def extract_surfaces(self, thres):
        """
        Extracts all the surfaces present in the field. SLOOOOW
        """
        labeled, nsurf = self.label_surfaces(thres)
        surflist = list()
        for i in range(1, nsurf+1):
            surflist.append(Surface(np.where(labeled == i), thres))

        return surflist

    def extract_largest_surface(self, thres):
        """
        Extracts the largest surface present in the field
        """
        labeled, nsurf = self.label_surfaces(thres)
        volumes = np.empty((nsurf,), dtype=np.int)

        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            vol = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])
            if vol > labeled.shape[0]*labeled.shape[2]:
                strides = np.where(labeled == objnum+1)
                logging.info(
                    'Found volume {} big. Stopping'.format(len(strides[0])))
                return Surface(strides, thres)

            volumes[objnum] = vol

        strides = np.where(labeled == volumes.argmax()+1)
        logging.info(
            'Found volume {} for {} total'.format(len(strides[0]),
                                                  np.prod(labeled.shape))
            )
        return Surface(strides, thres)

    def extract_surfaces_gt(self, thres, nvoxels=27):
        """
        Provides the surface list for surfaces larger than nvoxels. SLOOOW
        """
        labeled, nsurf = self.label_surfaces(thres)
        volumes = np.empty((nsurf,), dtype=np.int)
        mask = np.zeros(labeled.shape, dtype=np.bool)

        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            vol = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])
            if vol > nvoxels:
                mask = np.bitwise_or(mask, labeled == objnum+1)

        return Surface(np.where(mask), thres)

    def generate_target_points(self, thres, NUM, OFFSET):
        """
        Returns NUM random samples, framed with OFFSET, from the
        field. For distance computation.
        """
        nx, ny, nz = self.data.shape
        nx = nx-2*OFFSET
        nz = nz-2*OFFSET
        # print "...Framed shape", nx, ny, nz
        trgt = np.empty((NUM, 3), dtype=np.double)
        sval = np.empty((NUM,), dtype=np.double)
        side = np.empty((NUM,), dtype=np.int8)
        
        guessi = OFFSET + randint(0, nx-1, size=NUM)
        guessj = randint(0, ny-1, size=NUM)
        guessk = OFFSET + randint(0, nz-1, size=NUM)
        
        mask = self.label_gt_largest_mask(thres).astype(np.int8)

        for n in range(NUM):
            trgt[n, 0] = self.xr[guessi[n]]
            trgt[n, 1] = self.yr[guessj[n]]
            trgt[n, 2] = self.zr[guessk[n]]
            sval[n] = self.data[guessi[n], guessj[n], guessk[n]]
            #This is vertex centered, while the surface is center
            #centered, maybe you have to interpolate
            side[n] = 2*mask[guessi[n], guessj[n], guessk[n]]-1
                
        return trgt, sval, side

    def generate_weighted_points(self, thres, NUM, OFFSET):
        """
        Returns NUM random samples, framed with OFFSET, from the
        field. For distance computation.
        """
        nx, ny, nz = self.data.shape
        nx = nx-2*OFFSET
        nz = nz-2*OFFSET
        # print "...Framed shape", nx, ny, nz
        trgt = np.empty((NUM, 3), dtype=np.double)
        sval = np.empty((NUM,), dtype=np.double)
        side = np.empty((NUM,), dtype=np.int8)
        weight = np.empty((NUM,), dtype=np.double)
        
        dy = np.zeros((ny,), dtype=np.double)
        dy[:-1] = np.diff(self.yr)
        dy[-1] = dy[-2]
        
        guessi = OFFSET + randint(0, nx-1, size=NUM)
        guessj = randint(0, ny-1, size=NUM)
        guessk = OFFSET + randint(0, nz-1, size=NUM)
        
        mask = self.label_gt_largest_mask(thres).astype(np.int8)

        for n in range(NUM):
            trgt[n, 0] = self.xr[guessi[n]]
            trgt[n, 1] = self.yr[guessj[n]]
            trgt[n, 2] = self.zr[guessk[n]]
            sval[n] = self.data[guessi[n], guessj[n], guessk[n]]
            #This is vertex centered, while the surface is center
            #centered, maybe you have to interpolate
            side[n] = 2*mask[guessi[n], guessj[n], guessk[n]]-1
            weight[n] = dy[guessj[n]]
                
        return trgt, sval, side, weight


class TestField(unittest.TestCase):
    """
    Achtung! this tests both Field and Surface classes
    """
    def setUp(self):
        """
        Create a sphere and a torus
        """
        N = 32
        self.N = N  # Just in case
        xgrid = np.linspace(-2, 2, N)
        ygrid = np.linspace(-2, 2, N)
        zgrid = np.linspace(-2, 2, N)
        domain = np.empty((N, N, N), dtype=np.float32)
        for i, j, k in product(range(N), range(N), range(N)):
            domain[i, j, k] = np.sqrt(xgrid[i]**2 + ygrid[j]**2 + zgrid[k]**2)

        self.sphere = Field(domain[:, :, :], xgrid, ygrid, zgrid)  # Threshold 1.0

    def test_extract_largest_surface_sphere(self):
        sphere_surface = self.sphere.extract_largest_surface(1.0)
        self.assertEqual(sphere_surface.thres, 1.0)
        self.assertEqual(sphere_surface.genus(), -1)

    def test_point_list(self):
        sphere_surface = self.sphere.extract_largest_surface(1.0)
        points = sphere_surface.point_list(self.sphere)

        self.assertEqual(len(points), sphere_surface.number_of_voxels())
        #Compute the error of the method
        norm = ((points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)-1).mean()
        self.assertTrue(np.abs(norm) < 0.005, "Error higher than expected {}".format(np.abs(norm)))

    def test_refined_point_list(self):
        sphere_surface = self.sphere.extract_largest_surface(1.0)
        points = sphere_surface.refined_point_list(self.sphere)

        self.assertEqual(len(points), sphere_surface.number_of_voxels())
        # Compute the error of the method
        norm = ((points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)-1).mean()
        self.assertTrue(np.abs(norm) < 0.003,
                        "Error higher than expected {} > {}".format(norm, 0.003))

    def test_label_lt_largest(self):
        sphere_entity = self.sphere.label_lt_largest(1.0)
        self.assertEqual(sphere_entity.number_of_voxels(), 1904)
        self.assertEqual(sphere_entity.genus(), 0)

    def test_label_gt_largest(self):
        sphere_entity = self.sphere.label_gt_largest(1.0)
        self.assertEqual(sphere_entity.number_of_voxels(), 30864)
        self.assertEqual(sphere_entity.genus(), -1)
