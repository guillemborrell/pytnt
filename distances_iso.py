from __future__ import division,print_function
from itertools import product
from scipy.spatial import cKDTree
from numpy.random import rand
import tables
import time
import numpy as np
from field import Field
import pickle
import os

if __name__ == '__main__':
    pylab.close('all')
    hists = list()

    NTOT = 1024
    N = 512
    #Create the data
    f = tables.openFile('/data4/guillem/distances/BK0000_omom.h5','r')
    data = np.sqrt(f.root.Data[:N,:N,:N])
    data = data/np.sqrt(data.mean()**2+ data.std()**2)
    eta = f.root.Kolm_L.read()
    Rel = 433.0
    print('Kolmogorov scale',eta)

    x = np.linspace(0,2*np.pi*N/NTOT,N)/eta
    y = np.linspace(0,2*np.pi*N/NTOT,N)/eta
    z = np.linspace(0,2*np.pi*N/NTOT,N)/eta

    field = Field(data,x,y,z)

    FRAME = N//8

    thresholds = np.logspace(-0.6,0.3,5)
    for thres in thresholds:
        print('Computing the pdf for the threshold {}'.format(thres))
        surface = field.extract_complete_surface(thres)
        voxels = surface.oversampled_point_list(field)
        
        sval = data[FRAME:N-FRAME,FRAME:N-FRAME,FRAME:N-FRAME].reshape((N-2*FRAME)**3)
        side = (2*(np.sign(data > thres).astype(np.int32))-1)[FRAME:N-FRAME,FRAME:N-FRAME,FRAME:N-FRAME].reshape((N-2*FRAME)**3)
        trgt = np.empty(((N-2*FRAME)**3,3),dtype=np.double)
        
        count = 0
        for i,j,k in product(range(FRAME,N-FRAME),
                             range(FRAME,N-FRAME),
                             range(FRAME,N-FRAME)):
            trgt[count,0] = x[i]
            trgt[count,1] = y[j]
            trgt[count,2] = z[k]
            count += 1

        now = time.clock()
        t = cKDTree(voxels)
        print('Building the tree took {} seconds'.format(time.clock()-now))
        now = time.clock()
        dist = t.query(trgt)[0]*side
        print('Distance queries took {} seconds'.format(time.clock()-now))
        hists.append(np.histogram2d(dist,
                                    np.log10(sval),
                                    bins=500))

    for i,hist in enumerate(hists):
        fig = pylab.figure(i+1)
        semilogx(10**hist[2][1:],
                 np.dot(hist[0].T,hist[1][1:])/np.sum(hist[0],axis=0),
                 linewidth=2)
        histplot = np.log(hist[0]/hist[0].sum())
        contour(10**hist[2][1:],hist[1][1:],histplot/np.abs(histplot.max()))
        semilogx(thresholds[i]*np.ones(hist[2].shape),
                 hist[1],
                 'k--',linewidth=2)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(top=0.85)
        xlabel(r'$|\omega|/\omega^\prime$',fontsize=22)
        ylabel(r'$d/\eta$',fontsize=22)
        title(r'$Re_\lambda={}$'.format(Rel),fontsize=22)
        xlim([0.02,15])
        ylim([-60,10])

    f.close()
