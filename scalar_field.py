from field import Field
from itertools import product
from scipy import interpolate
from lkdtree import distances
import numpy as np


class VorticityMagnitudeField(Field):
    """
    This class starts to be really specific. It is intended to
    be used with scalar fields of wall bounded turbulent flows
    like enstrophy, vorticity magnitude, vorticity components...
    """
    def __init__(self, data, stats, NX0):
        """
        Data field
        """
        stats.read()
        self.stats = stats
        x = stats.x[:data.shape[0]]
        # It assumes that the first point of the dataset in the
        # wall normal direction is the wall.
        y = stats.yr[:data.shape[1]]
        z = stats.zr[:data.shape[2]]
        super(VorticityMagnitudeField, self).__init__(data, x, y, z)
        self.NX0 = NX0

    @property
    def ydelta(self):
        """
        Return y/delta99 at the middle of the box
        """
        return self.yr/self.stats.delta99(self.NX0+self.NX/2)

    def scale_factor_wall(self):
        """
        Returns the scale factor for vorticity magnitude in wall units
        """
        NX = self.data.shape[0]
        return self.stats.Re * self.stats.utau[self.NX0 + NX/2]**2

    def scale_factor_outer(self):
        """
        Returns the scale factor for vorticity magnitude in outer units
        """
        NX = self.data.shape[0]
        return (self.stats.Re * self.stats.utau[self.NX0 + NX/2]**2)/np.sqrt(
            self.stats.Retau(self.NX0 + NX/2))

    def scale_wall(self):
        """
        Scale the vorticity magnitude in wall units. Changes the data array
        """
        NX = self.data.shape[0]

        for i in range(NX):
            adim = self.stats.Re * self.stats.utau[self.NX0 + i]**2
            self.data[i, :, :] = self.data[i, :, :]/adim

    def scale_outer(self):
        """
        Scale the vorticity magnitude in outer units. Changes the data array
        """
        NX = self.data.shape[0]

        for i in range(NX):
            adim = (self.stats.Re * self.stats.utau[self.NX0 + i]**2)/np.sqrt(
                self.stats.Retau(self.NX0 + i))
            self.data[i, :, :] = self.data[i, :, :]/adim

    def intermittency_profile(self, thres):
        """
        Intermittency profile in the non homogeneous (second)
        direction.
        """
        (NX, NY, NZ) = self.data.shape
        result = np.empty((NY,), dtype=np.double)
        for j in range(NY):
            result[j] = np.array(np.count_nonzero(
                self.data[:,j,:] > thres), dtype=np.double)/NX/NZ

        return result

    def kolmogorov_length_at_height(self, height=0.6):
        NX0 = self.NX0
        NX = self.data.shape[0]
        h = np.where(self.stats.ydelta(NX0 + NX/2) > height)[0][0]
        return self.stats.dissipation()[NX0 + NX/2, h]**(-0.25) / \
            self.stats.utau[NX0 + NX/2]/self.stats.Re

    def taylor_microscale_at_height(self, height=0.6):
        NX0 = self.NX0
        NX = self.data.shape[0]
        h = np.where(self.stats.ydelta(NX0 + NX/2) > height)[0][0]
        np.sqrt(5*(
            self.stats.us[NX0 + NX/2, h]**2 +
            self.stats.vs[NX0 + NX/2, h]**2 +
            self.stats.ws[NX0 + NX/2, h]**2
        )/self.stats.Re/(self.stats.dissipation()[NX0 + NX/2, h] *
            self.stats.Re * self.stats.utau[NX0 + NX/2]**4)
        )

    def vertical_distance_profile(self, thres):
        """
        Vertical distance profile since first detection of the threshold.
        This is the usual method found in the bibliography.
        """
        RANGE = 0.5
        NX = self.data.shape[0]
        NZ = self.data.shape[2]

        #Coordinates at the vertices
        yr=self.yr.copy()[::-1]
        yr[:] = -(yr[:]-yr[0])
        ogrid = np.linspace(-RANGE,RANGE,100)
        acc = np.zeros((100,),dtype=np.float64)
        data = np.zeros((len(yr),),dtype=np.float32)

        for i,k in product(range(NX),range(NZ)):
            data[:] = self.data[i,::-1,k] 
            yloc = yr[np.where(data>thres)[0][0]]
            itp = interpolate.interp1d(yr-yloc,data)
            acc[:] = acc[:] + itp(ogrid)
    
        return acc/NX/NZ

    def vertical_distance_histogram(self, thres, ybins, xbins):
        """
        Vertical distance histogram from the first vertical detection.
        """
        RANGE = 0.1
        NX = self.data.shape[0]
        NY = self.data.shape[1]
        NZ = self.data.shape[2]

        yr = self.yr.copy()[::-1]
        yr[:] = -(yr[:]-yr[0])
        ogrid = np.linspace(-RANGE, RANGE, 100)
        acc = np.zeros((100,), dtype=np.float64)
        data = np.zeros((len(yr),),dtype=np.float32)
        height_map = np.empty((NX, NZ), dtype=np.float64)

        #First thing is to compute the height map
        for i, k in product(range(NX), range(NZ)):
            data[:] = self.data[i,::-1,k]
            yloc = yr[np.where(data > thres)[0][0]]
            itp = interpolate.interp1d(yr-yloc, data)
            acc[:] = itp(ogrid)
            height_map[i, k] = -ogrid[np.where(acc > thres)[0][0]] - yloc + yr[-1]

        # Set histogram up
        NHISTY = len(ybins)
        NHISTX = len(xbins)
        histogram = np.zeros((NHISTY, NHISTX), dtype=np.double)

        for i, j, k in product(range(NX), range(NY), range(NZ)):
            histogram[np.where(ybins > self.y[j])[0][0],
                      np.where(xbins > np.log10(self.data[i, j, k]))[0][0]] += 1.0

        return histogram

    def ball_distance_histogram(self, thres, nbins=200, npoints=1000000, FRAME=100):
        """
        Minimum ball distance histogram from the single largest surface.
        """
        surface = self.extract_largest_surface(thres)
        voxels = surface.refined_point_list(self)
        trgt, sval = self.generate_target_points(npoints, FRAME)
        dist = distances(voxels, trgt)
        return np.histogram2d(dist, np.log10(sval), bins=nbins)


class VorticityComponentField(VorticityMagnitudeField):
    def scale_factor_outer(self):
        """
        Returns the scale factor for vorticity magnitude in outer units
        """
        NX = self.data.shape[0]
        return self.stats.Re * self.stats.utau[self.NX0 + NX/2]**2 *\
            np.sqrt(self.stats.Retau()[self.NX0 + NX/2]) / np.sqrt(3)

    def scale_factor_wall(self):
        """
        Returns the scale factor for vorticity magnitude in wall units
        """
        NX = self.data.shape[0]
        return self.stats.Re * self.stats.utau[self.NX0 + NX/2]**2 / np.sqrt(3)

    def scale_wall(self):
        """
        Scale the vorticity magnitude in wall units. Changes the data array
        """
        NX = self.data.shape[0]

        for i in range(NX):
            adim = self.stats.Re * self.stats.utau[self.NX0 + i]**2 / np.sqrt(3)
            self.data[i, :, :] = self.data[i, :, :]/adim

    def scale_outer(self):
        """
        Scale the vorticity magnitude in outer units. Changes the data array
        """
        NX = self.data.shape[0]

        for i in range(NX):
            adim = (self.stats.Re * self.stats.utau[self.NX0 + i]**2)/np.sqrt(
                self.stats.Retau(self.NX0 + i)) / np.sqrt(3)
            self.data[i, :, :] = self.data[i, :, :]/adim
