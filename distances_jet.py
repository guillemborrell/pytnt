from __future__ import division,print_function
from itertools import product
from scipy.spatial import cKDTree
from numpy.random import rand
import tables
import time
import numpy as np
from field import Field
from loadjet import loadjet
import pickle
import os

if __name__ == '__main__':
    pylab.close('all')
    hists = list()
    kol_eta = 4.279E-3

    z,y,x,data = loadjet('/data4/guillem/jet/wabs_yi_3D_3702_data.bin')
    central_plane = data[:,y.shape[0]/2,:]
    data = data[:128,500:-500,:256]/np.sqrt(
        central_plane.mean()**2 + central_plane.std()**2)
    x = x[:128].astype(np.double)/kol_eta
    y = y[500:-500].astype(np.double)/kol_eta
    z = z[:256].astype(np.double)/kol_eta

    NX = len(x)
    NY = len(y)
    NZ = len(z)

    field = Field(data,x,y,z)

    thresholds = np.logspace(-1,0,5)
    for thres in thresholds:
        print('Computing the pdf for the threshold {}'.format(thres))
        surface = field.extract_complete_surface(thres)
        voxels = surface.refined_point_list(field)
        
        sval = data.reshape(NX*NY*NZ)
        side = (2*(np.sign(data > thres).astype(np.int32))-1).reshape(NX*NY*NZ)
        trgt = np.empty((NX*NY*NZ,3),dtype=np.double)
        
        count = 0
        for i,j,k in product(range(NX),range(NY),range(NZ)):
            trgt[count,0] = x[i]
            trgt[count,1] = y[j]
            trgt[count,2] = z[k]
            count += 1

        now = time.clock()
        t = cKDTree(voxels)
        print('Building the tree took {} seconds'.format(time.clock()-now))
        now = time.clock()
        dist = t.query(trgt)[0]*side.astype(np.double)
        print('Distance queries took {} seconds'.format(time.clock()-now))
        hists.append(np.histogram2d(dist,
                                    np.log10(sval),
                                    bins=500))

    for i,hist in enumerate(hists):
        fig = pylab.figure(i+1)
        semilogx(10**hist[2][1:],
                 np.dot(hist[0].T,hist[1][1:])/np.sum(hist[0],axis=0),
                 linewidth=2)
        contour(10**hist[2][1:],hist[1][1:],np.log(hist[0]))
        semilogx(thresholds[i]*np.ones(hist[2].shape),
                 hist[1],
                 'k--',linewidth=2)
        fig.subplots_adjust(bottom=0.15)
        xlabel(r'$\omega/\omega^\prime$',fontsize=22)
        ylabel(r'$d$',fontsize=22)

    
