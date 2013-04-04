from __future__ import print_function
import numpy as np

#Modules imported for testing
import unittest
from itertools import product, count, izip


class Entity(object):
    """
    Class that models any geometric entity made from voxels
    """
    def __init__(self, voxels):
        self.voxels = voxels

    def number_of_voxels(self):
        return len(self.voxels[0])

    def box_counting_exponent(self):
        """
        Computes the box counting exponent, that is used to compute the
        fractal dimension. It can be used in non cubic boxes.
        """
        NY = len(self.voxels[1])
        NZ = len(self.voxels[2])

        # The implementation uses a hash table to identify voxels in the tree
        boxcount = list()
        value = np.arange(len(self.voxels[0]))
        boxcount.append(len(value))

        level = 1
        while len(value) > 1:
            x = self.voxels[0][value]/2**level
            y = self.voxels[1][value]/2**level
            z = self.voxels[2][value]/2**level
            idx = np.unique(x*NY*NZ + y*NZ + z,return_index=True)[1]
            value = value[idx]
            
            level += 1
            boxcount.append(len(value))

        return np.array(boxcount)

    def genus(self):
        """
        Compute the genus of the polyhedron formed by the voxel group.
        """
        #Compute the size of the test domain for the genus.
        NX = self.voxels[0].max()
        NY = self.voxels[1].max()
        NZ = self.voxels[2].max()
        domain = np.zeros((NX+3, NY+3, NZ+3), dtype=np.bool)

        #Create a domain with the appropiate size with the given voxels.
        domain[self.voxels] = True
        domain[1:, 1:, 1:] = domain[:-1, :-1, :-1]

        # Compute the Euler characteristic
        # The algorithm goes as follows:
        #
        #  * Check every vertex of the domain if it is internal or external.
        #    An external vertex will not be completely surrunded by voxels.
        #    Count every external verted
        #
        #  * Check every edge the same way. For convenience, the edges of
        #    different orientations are checked separately and added together
        #
        #  * Check every face. For convenience, faces of different orientations
        #    are checked separately.
        #
        #  * Compute the Euler characteristic with the usual formula.
        
        # Number of total vertices of the polyhedron
        nvertices = np.logical_xor(
            np.bitwise_or(
                domain[:-1, :-1, :-1],
                np.bitwise_or(
                    domain[:-1, :-1, 1:],
                    np.bitwise_or(
                        domain[:-1, 1:, :-1],
                        np.bitwise_or(
                            domain[:-1, 1:, 1:],
                            np.bitwise_or(
                                domain[1:, :-1, :-1],
                                np.bitwise_or(
                                    domain[1:, :-1, 1:],
                                    np.bitwise_or(
                                        domain[1:, 1:, :-1],
                                        domain[1:, 1:, 1:])
                                    )
                                )
                            )
                        )
                    )
                ),
            np.bitwise_and(
                domain[:-1, :-1, :-1],
                np.bitwise_and(
                    domain[:-1, :-1, 1:],
                    np.bitwise_and(
                        domain[:-1, 1:, :-1],
                        np.bitwise_and(
                            domain[:-1, 1:, 1:],
                            np.bitwise_and(
                                domain[1:, :-1, :-1],
                                np.bitwise_and(
                                    domain[1:, :-1, 1:],
                                    np.bitwise_and(
                                        domain[1:, 1:, :-1],
                                        domain[1:, 1:, 1:])
                                    )
                                )
                            )
                        )
                    )
                )).sum()
            
        ##### EDGES
        #edgesx
        nedges = np.logical_xor(
            np.logical_or(
                domain[:, :-1, :-1],
                np.logical_or(
                    domain[:, :-1, 1:],
                    np.logical_or(
                        domain[:, 1:, :-1],
                        domain[:, 1:, 1:])
                    )
                ),
            np.logical_and(
                domain[:, :-1, :-1],
                np.logical_and(
                    domain[:, :-1, 1:],
                    np.logical_and(
                        domain[:, 1:, :-1],
                        domain[:, 1:, 1:])
                            )
                        )
                    ).sum()


        #edgesy
        nedges += np.logical_xor(
            np.logical_and(
                domain[:-1, :, :-1],
                np.logical_and(
                    domain[:-1, :, 1:],
                    np.logical_and(
                        domain[1:, :, :-1],
                        domain[1:, :, 1:])
                    )
                ),
            np.logical_or(
                domain[:-1, :, :-1],
                np.logical_or(
                    domain[:-1, :, 1:],
                    np.logical_or(
                        domain[1:, :, :-1],
                        domain[1:, :, 1:])
                    )
                )
            ).sum()
            
        #edgesz
        nedges += np.logical_xor(
            np.logical_and(
                domain[:-1, :-1, :],
                np.logical_and(
                    domain[:-1, 1:, :],
                    np.logical_and(
                        domain[1:, :-1, :], 
                        domain[1:, 1:, :])
                    )
                ),
            np.logical_or(
                domain[:-1, :-1, :],
                np.logical_or(
                    domain[:-1, 1:, :],
                    np.logical_or(
                        domain[1:, :-1, :], 
                        domain[1:, 1:, :])
                    )
                )
            ).sum()
            
        #Faces
            
        #### FACES
        # faces x
        nfaces = np.logical_xor(
            np.logical_and(domain[:-1, :, :], domain[1:, :, :]),
            np.logical_or(domain[:-1, :, :], domain[1:, :, :])
            ).sum()
                    
        # faces y
        nfaces += np.logical_xor(
            np.logical_and(domain[:, :-1, :], domain[:, 1:, :]),
            np.logical_or(domain[:, :-1, :], domain[:, 1:, :])
            ).sum()
        
        # faces z
        nfaces += np.logical_xor(
            np.logical_and(domain[:, :, :-1], domain[:, :, 1:]),
            np.logical_or(domain[:, :, :-1], domain[:, :, 1:])
            ).sum()
        
        xi = nvertices-nedges+nfaces #Euler characteristic
        g = (2-xi)/2 #genus

        return g


class TestEntity(unittest.TestCase):
    def setUp(self):
        """
        Create a sphere and a torus
        """
        N = 32
        xgrid = np.linspace(-2, 2, N)
        ygrid = np.linspace(-2, 2, N)
        zgrid = np.linspace(-2, 2, N)
        domain = np.empty((N, N, N), dtype=np.double)
        for i, j, k in product(range(N), range(N), range(N)):
            domain[i, j, k] = xgrid[i]**2 + ygrid[j]**2 + zgrid[k]**2

        self.nnz_sphere = np.count_nonzero(domain < 1.0)
        self.sphere = Entity(np.where(domain < 1.0))

        for i, j, k in product(range(N), range(N), range(N)):
            domain[i, j, k] = (1-np.sqrt(xgrid[i]**2+ygrid[j]**2))**2+zgrid[k]**2

        self.nnz_torus = np.count_nonzero(domain < 0.2)
        self.torus = Entity(np.where(domain < 0.2))

    def test_number_of_voxels_sphere(self):
        self.assertEqual(self.sphere.number_of_voxels(), self.nnz_sphere)

    def test_number_of_voxels_torus(self):
        self.assertEqual(self.torus.number_of_voxels(), self.nnz_torus)

    def test_genus_sphere(self):
        # Because everybody knows that the genus of a sphere is 0
        self.assertEqual(self.sphere.genus(), 0)

    def test_genus_torus(self):
        # The genus of a torus is 1 because it has 1 single handle
        self.assertEqual(self.torus.genus(), 1)

    def test_box_counting(self):
        nboxes = self.sphere.box_counting_exponent()
        self.assertTrue((nboxes[-6:] == np.array([1904, 304, 56, 8, 8, 1])).all(), "{}".format(nboxes))
