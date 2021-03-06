from __future__ import division, print_function

from field import Field
from itertools import product
from scipy import interpolate
from scipy.spatial import cKDTree
import logging
import time
import numpy as np
from histogram3d import histogram3d
from numpy.random import randint


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
        Scale the vorticity magnitude in wall units. Changes the data
        array
        """
        NX = self.data.shape[0]

        for i in range(NX):
            adim = self.stats.Re * self.stats.utau[self.NX0 + i]**2
            self.data[i, :, :] = self.data[i, :, :]/adim

    def scale_outer(self):
        """
        Scale the vorticity magnitude in outer units. Changes the data
        array
        """
        NX = self.data.shape[0]

        for i in range(NX):
            adim = (self.stats.Re * self.stats.utau[self.NX0 + i]**2)/np.sqrt(
                self.stats.Retau(self.NX0 + i))
            self.data[i, :, :] = self.data[i, :, :]/adim

    def scale_inner(self):
        """
        Scale the vorticity magnitude in turbulent units. Changes the
        data array
        """
        NX = self.data.shape[0]
        NY = self.data.shape[1]

        for i in range(NX):
            adim = (self.stats.Re * self.stats.utau[self.NX0 + i]**2)/np.sqrt(
                self.stats.Retau(self.NX0 + i))
            self.data[i, :, :] = self.data[i, :, :]/adim

            for j in range(1,NY):
                adim = self.yr[j]/self.stats.delta99(self.NX0+i)
                self.data[i, j, :] = self.data[i, j, :]/adim**(-1/2)

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

        return np.sqrt(5*(
            self.stats.us[NX0 + NX/2, h]**2 +
            self.stats.vs[NX0 + NX/2, h]**2 +
            self.stats.ws[NX0 + NX/2, h]**2
        )/self.stats.Re/(self.stats.dissipation()[NX0 + NX/2, h] *
            self.stats.Re * self.stats.utau[NX0 + NX/2]**4)
        )

    def interface_height_map(self, thres):
        """
        Computes the height map of the interface. Useful to further
        compute the vertical distances or to analyze the interface in
        a glance.
        """
        NX = self.data.shape[0]
        NY = self.data.shape[1]
        NZ = self.data.shape[2]

        yr = self.yr.copy()[::-1]
        yr[:] = -(yr[:]-yr[0])
        data = np.zeros((len(yr),),dtype=np.float32)
        height_map = np.zeros((NX, NZ), dtype=np.float64)

        for i, k in product(range(NX), range(NZ)):
            data[:] = self.data[i,::-1,k]
            ylocidx = np.where(data > thres)[0][0]
            datab = -(data[ylocidx]-thres)
            ylocb = yr[ylocidx]
            datat = data[ylocidx-1]-thres
            yloct = yr[ylocidx-1]
            height_map[i, k] = (yloct*datat + ylocb*datab)/(datab + datat)

        return yr[-1]-height_map

    def cleaned_interface_height_map(self, thres):
        """
        Computes the height map of the interface. Useful to further
        compute the vertical distances or to analyze the interface in
        a glance.
        """
        #This is correct, because the mask is at the vertices of the
        #voxels.
        mask = self.label_gt_largest_mask(thres)
        NX = self.data.shape[0]
        NY = self.data.shape[1]
        NZ = self.data.shape[2]

        yr = self.yr.copy()[::-1]
        yr[:] = -(yr[:]-yr[0])
        data = np.zeros((len(yr),),dtype=np.float32)
        height_map = np.zeros((NX, NZ), dtype=np.float64)

        for i, k in product(range(NX), range(NZ)):
            data[:] = self.data[i,::-1,k]
            lmask = mask[i,::-1,k]
            ylocidx = np.where(lmask == True)[0][0]
            datab = -(data[ylocidx]-thres)
            ylocb = yr[ylocidx]
            datat = data[ylocidx-1]-thres
            yloct = yr[ylocidx-1]
            height_map[i, k] = (yloct*datat + ylocb*datab)/(datab + datat)

        return yr[-1]-height_map

    def vertical_distance_profile(self, thres, RANGE=0.5):
        """
        Vertical distance profile since first detection of the threshold.
        This is the usual method found in the bibliography.
        """
        NOUT = 200
        hmap = self.interface_height_map(thres)
        NX = self.data.shape[0]
        NZ = self.data.shape[2]
        ogrid = np.linspace(-RANGE, RANGE, NOUT)
        res = np.zeros((NOUT,), dtype=np.double)

        for i,k in product(range(NX),range(NZ)):
            itp = interpolate.interp1d(self.yr-hmap[i, k],
                                       self.data[i,:,k],
                                       kind='linear')
            res += itp(ogrid)

        return res/(NX*NZ), ogrid

    def vertical_distance_histogram(self, thres, nbins=200,
                                    clean=False, scale='log'):
        if clean:
            hmap = self.cleaned_interface_height_map(thres)
        else:
            hmap = self.interface_height_map(thres)
        NX = self.data.shape[0]
        NY = self.data.shape[1]
        NZ = self.data.shape[2]
        
        dist = np.zeros((NX, NY, NZ), dtype=np.double)
        for i,k in product(range(NX), range(NZ)):
            dist[i,:,k] = hmap[i,k] - self.yr

        if scale == 'rect':
            data = self.data.reshape(NX*NY*NZ)
        else:
            data = np.log10(self.data).reshape(NX*NY*NZ)

        return np.histogram2d(dist.reshape(NX*NY*NZ),
                              data,
                              bins=nbins)

    def vertical_distance_weighted_histogram(self, thres, nbins=200,
                                             clean=False, scale='log'):
        """
        Weighted vertical histogram, hopefully a pdf.
        
        The memory usage for this case in large fields is humongous,
        this is the reason why I decided not to use all the columns in
        the field. Come on, it should be the same.
        """
        if clean:
            hmap = self.cleaned_interface_height_map(thres)
        else:
            hmap = self.interface_height_map(thres)
        NX = self.data.shape[0]
        NY = self.data.shape[1]
        NZ = self.data.shape[2]
        
        dy = np.zeros((NY,), dtype=np.double)
        dy[:-1] = np.diff(self.yr)
        dy[-1] = dy[-2]

        dist = np.zeros((NX/2, NY, NZ/2), dtype=np.double)
        weight = np.zeros((NX/2, NY, NZ/2), dtype=np.double)
        
        for i,k in product(range(NX), range(NZ)):
            dist[i/2,:,k/2] = hmap[i,k] - self.yr
            weight[i/2,:,k/2] = dy

        if scale == 'rect':
            data = self.data[::2,:,::2].reshape(NX*NY*NZ/4)
        else:
            data = np.log10(self.data[::2,:,::2]).reshape(NX*NY*NZ/4)

        return np.histogram2d(dist.reshape(NX*NY*NZ/4),
                              data,
                              bins=nbins,
                              weights=weight.reshape(NX*NY*NZ/4))

    def vertical_distance_histogram3d(self, thres, nbins=200):
        """
        3D histogram of the magnitude, the vertical distance to the
        surface, and the vertical distance to the wall.
        """
        hmap = self.interface_height_map(thres)
        NX = self.data.shape[0]
        NY = self.data.shape[1]
        NZ = self.data.shape[2]

        #Project the bins
        #Bins for the magnitude
        minmag = np.log10(self.data.min())
        maxmag = np.log10(self.data.max())
        binsmag = np.linspace(minmag, maxmag, nbins)

        #Bins for distance to the surface
        mindist = 0.0
        maxdist = np.max(np.abs(np.array([self.yr[-1]-hmap.min(), hmap.max()])))
        binsdist = np.linspace(mindist, maxdist, nbins)

        #Bins for distance to the wall
        minheight = 0.0
        maxheight = self.yr[-1]
        binsheight = np.linspace(minheight, maxheight, nbins)
        
        histogram = histogram3d(binsmag, binsdist, binsheight)

        for i,k in product(range(NX), range(NZ)):
            histdata = np.array([np.log10(self.data[i,:,k]),
                                 np.abs(self.yr-hmap[i,k]),
                                 self.yr])
            histogram.increment(histdata)
            
        return histogram
    
    def ball_distance_mask(self,thres,FRAME):
        mask = self.label_gt_largest_mask(thres).astype(np.int8)
        return 2*mask-1


    def ball_distance_field(self,thres,FRAME):
        surface = self.extract_largest_surface(thres)
        voxels = surface.refined_point_list(self)
        now = time.clock()
        t = cKDTree(voxels)
        logging.info(
            'Building the tree took {} s.'.format(time.clock()-now))
        
        nx,ny,nz = self.data.shape
        nx = nx-2*OFFSET
        nz = nz_2*OFFSET
        
        field = np.empty((nx,ny,nz),dtype=np.double)
        points = np.empty((ny,3), dtype= np.double)
        side = self.ball_distance_mask(thres,FRAME)[OFFSET:-OFFSET,
                                                    :,
                                                    OFFSET:-OFFSET]

        for i,k in product(range(nx),range(nz)):
            if k == 0: logging.info('{} of {}'.format(i,nx))
            points[:,0] = self.xr[i]
            points[:,1] = self.yr
            points[:,2] = self.zr[k]
            field[i,:,k] = t.query(points)[0]*side[i,:,k]

        return field


    def ball_distance_histogram(self, thres, nbins=200,
                                npoints=1000000,
                                FRAME=100, scale='log'):
        """
        Minimum ball distance histogram from the single largest surface.
        """
        surface = self.extract_largest_surface(thres)
        voxels = surface.refined_point_list(self)
        trgt, sval, side= self.generate_target_points(
            thres, npoints, FRAME)
        now = time.clock()
        t = cKDTree(voxels)
        logging.info(
            'Building the tree took {} s.'.format(time.clock()-now))
        now = time.clock()
        dist = t.query(trgt)[0]*side
        logging.info('Distances took {} s'.format(time.clock()-now))
        now = time.clock()

        if scale == 'rect':
            pass
        elif scale == 'log':
            sval = np.log10(sval)
        else:
            raise ValueError("scale not 'rect' or 'log'")

        res = np.histogram2d(dist, sval, bins=nbins)
        logging.info('Histogram {} s'.format(time.clock()-now))
        return res

    def ball_distance_weighted_histogram(self, thres, nbins=200, npoints=1000000,
                                         FRAME=100, scale='log'):
        """
        Minimum ball distance histogram from the single largest surface.
        """
        surface = self.extract_largest_surface(thres)
        voxels = surface.refined_point_list(self)
        trgt, sval, side, weight = self.generate_weighted_points(thres, npoints, FRAME)
        now = time.clock()
        t = cKDTree(voxels)
        logging.info('Building the tree took {} s.'.format(time.clock()-now))
        now = time.clock()
        dist = t.query(trgt)[0]*side
        logging.info('Distances took {} s'.format(time.clock()-now))
        now = time.clock()

        if scale == 'rect':
            pass
        elif scale == 'log':
            sval = np.log10(sval)
        else:
            raise ValueError("scale not 'rect' or 'log'")

        res = np.histogram2d(dist, sval, bins=nbins, weights=weight)
        logging.info('Histogram {} s'.format(time.clock()-now))
        return res

    def ball_gradient_histogram(self, thres, nbins=200, npoints=1000000, FRAME=100):
        """
        Minimum ball distance histogram from the single largest surface.
        """
        surface = self.extract_largest_surface(thres)
        voxels = surface.refined_point_list(self)
        trgt, sval, side  = self.generate_target_points(thres, npoints, FRAME)
        sval = np.log10(np.abs(sval - thres)) #This makes the trick
        now = time.clock()
        t = cKDTree(voxels)
        logging.info('Building the tree took {} s.'.format(time.clock()-now))
        now = time.clock()
        dist = t.query(trgt)[0]*side
        logging.info('Distances took {} s'.format(time.clock()-now))
        now = time.clock()
        res = np.histogram2d(dist, sval/dist, bins=nbins)
        logging.info('Histogram {} s'.format(time.clock()-now))
        return res

    def ball_gradient_weighted_histogram(self, thres, nbins=200, npoints=1000000, FRAME=100):
        """
        Minimum ball distance histogram from the single largest surface.
        """
        surface = self.extract_largest_surface(thres)
        voxels = surface.refined_point_list(self)
        trgt, sval, side, weight  = self.generate_weighted_points(thres, npoints, FRAME)
        sval = np.log10(np.abs(sval - thres)) #This makes the trick
        now = time.clock()
        t = cKDTree(voxels)
        logging.info('Building the tree took {} s.'.format(time.clock()-now))
        now = time.clock()
        dist = t.query(trgt)[0]*side
        logging.info('Distances took {} s'.format(time.clock()-now))
        now = time.clock()
        res = np.histogram2d(dist, sval/dist, bins=nbins, weights=weight)
        logging.info('Histogram {} s'.format(time.clock()-now))
        return res


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


class VorticityMagnitudeFieldOffset(VorticityMagnitudeField):
    def __init__(self, data, stats, NX0, NY0):
        """
        Data field
        """
        super(VorticityMagnitudeFieldOffset, self).__init__(data, stats, NX0)
        self.NY0 = NY0

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
        guessj = randint(self.NY0, ny-1, size=NUM)
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
