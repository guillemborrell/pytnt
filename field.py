from __future__ import print_function
import numpy as np
from scipy.ndimage.measurements import label, find_objects
from entity import Entity
from surface import Surface
from numpy.random import randint

#imports for testing
import unittest
from itertools import product


class Field(object):
    def __init__(self, data, xr, yr, zr):
        self.data = data
        self.xr = xr
        self.yr = yr
        self.zr = zr

    def label_gt_largest(self, thres):
        """
        Extracts all the connected entities from the flow. Higher than
        the threshold. Fast.
        """
        labeled, nsurf = label(self.data > thres)

        volumes = np.empty((nsurf,), dtype=np.int)

        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])

        return Entity(np.where(labeled == volumes.argmax()+1))

    def label_lt_largest(self, thres):
        """
        Extracts all the connected entities from the flow. Lower than
        the threshold
        """
        labeled, nsurf = label(self.data < thres)

        volumes = np.empty((nsurf,), dtype=np.int)

        for obj, objnum in zip(find_objects(labeled), range(nsurf)):
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])

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
            volumes[objnum] = np.count_nonzero(labeled[obj[0], obj[1], obj[2]])

        return Surface(np.where(labeled == volumes.argmax()+1), thres)

    def extract_surfaces_gt(self, thres, nvoxels=27):
        """
        Provides the surface list for surfaces larger than nvoxels. SLOOOW
        """
        new_surface_list = list()

        surface_list = self.extract_surfaces(thres)

        for surface in surface_list:
            surfsize = surface.number_of_voxels()
            if surfsize > nvoxels:
                new_surface_list.append(surface)

        return new_surface_list

    def generate_target_points(self, NUM, OFFSET):
        """
        Returns NUM random samplessamples, framed with OFFSET, from
        the field. For distance computation.
        """
        nx, ny, nz = self.data.shape
        nx = nx-2*OFFSET
        nz = nz-2*OFFSET
        # print "...Framed shape", nx, ny, nz
        trgt = np.empty((NUM, 3), dtype=np.double)
        sval = np.empty((NUM,), dtype=np.double)

        guessi = OFFSET + randint(0, nx-1, size=NUM)
        guessj = randint(0, ny-1, size=NUM)
        guessk = OFFSET + randint(0, nz-1, size=NUM)

        for n in range(NUM):
            trgt[n, 0] = self.xr[guessi[n]]
            trgt[n, 1] = self.yr[guessj[n]]
            trgt[n, 2] = self.zr[guessk[n]]
            sval[n] = self.data[guessi[n], guessj[n], guessk[n]]

        return trgt, sval


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
        domain = np.empty((N, N, N), dtype=np.double)
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

    def test_label_lt(self):
        sphere_entity = self.sphere.label_lt(1.0)[0]  # Only one entity, I know
        self.assertEqual(sphere_entity.number_of_voxels(), 1904)
        self.assertEqual(sphere_entity.genus(), 0)

    def test_label_gt(self):
        sphere_entity = self.sphere.label_gt(1.0)[0]  # Only one entity, I know
        self.assertEqual(sphere_entity.number_of_voxels(), 30864)
        self.assertEqual(sphere_entity.genus(), -1)
