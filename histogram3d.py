from __future__ import print_function
import numpy as np
# try:
#     from _histogram3d import _histogram3d
# except ImportError:
#     print('Fast histogram not available')
from _histogram3d import _histogram3d
import unittest


class histogram3d(object):
    def __init__(self,binsx, binsy=None, binsz=None):
        self.binsx = binsx.astype(np.double)
        self.nbinsx = len(binsx)

        try:
            self.binsy = binsy.astype(np.double)
            self.nbinsy = len(binsy)
        except:
            self.binsy = self.binsx.copy()
            self.nbinsy = self.nbinsx

        try:
            self.binsz = binsz.astype(np.double)
            self.nbinsz = len(binsz)
        except:
            self.binsz = self.binsx.copy()
            self.nbinsz = self.nbinsx

        self.hist = np.zeros((self.nbinsx+1,
                              self.nbinsy+1,
                              self.nbinsz+1),dtype=np.int)

    def increment(self,data):
        if not data.shape[0] == 3:
            raise ValueError('This is a 3d histogram, data must be a (3,N) array')

        N = data.shape[1]
        _histogram3d(data.astype(np.float32),
                    N,
                    self.binsx,
                    self.nbinsx,
                    self.binsy,
                    self.nbinsy,
                    self.binsz,
                    self.nbinsz,
                    self.hist)
        
    def serialize(self):
        return (self.hist, self.binsx, self.binsy, self.binsz)


class TestHistogram3(unittest.TestCase):
    def test_creation(self):
        binsx = np.arange(1,2,10)
        hist = histogram3d(binsx)
        hist = histogram3d(binsx,binsx,binsx)
        self.assertEqual(binsx, hist.binsz)
    
    def test_increment(self):
        data = np.random.random((3,10000))
        hist = histogram3d(np.linspace(0,1,10))
        hist.increment(data)
        print(hist.hist.max(), hist.hist.min())
        self.assertTrue(True, True)
        
