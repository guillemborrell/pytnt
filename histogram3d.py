from __future__ import print_function
import numpy as np
try:
    from _histogram3d import _histogram3d
except ImportError:
    print('Fast histogram not available')
import unittest


class histogram3d(object):
    def __init__(self,binsx, binsy=False, binsz=False):
        self.binsx = binsx.astype(np.double)
        self.nbinsx = len(binsx)

        if binsy:
            self.binsy = binsy.astype(np.double)
            self.nbinsy = len(binsy)
        else:
            self.binsy = self.binsx.copy()
            self.nbinsy = self.nbinsx

        if binsz:
            self.binsz = binsz.astype(np.double)
            self.nbinsz = len(binsz)
        else:
            self.binsz = self.binsx.copy()
            self.nbinsz = self.nbinsx

        self.hist = np.zeros((self.nbinsx+1,
                              self.nbinsy+1,
                              self.nbinsz+1),dtype=np.int)

    def increment(self,data):
        if not data.shape[0] == 3:
            raise ValueError('This is a 3d histogram, data must be a (3,N) array')

        N = data.shape[1]
        _histogram3(data.astype(np.float32),
                    N,
                    self.binsx,
                    self.nbinsx,
                    self.binsy,
                    self.nbinsy,
                    self.binsz,
                    self.nbinsz,
                    self.hist)


class TestHistogram3(unittest.TestCase):
    def test_creation(self):
        binsx = np.arange(0,1,10)
        hist = histogram3(binsx)
        hist = histogram3(binsx,binsx,binsx)
        self.assertEqual(binsx, hist.binsz)
    
    def test_increment(self):
        data = np.random.rand(3,100)
        hist = histogram3(np.arange(0,1,10))
        hist.increment(data)

        self.assertTrue(True, True)
        
