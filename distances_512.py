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

    #Create the data
    f = tables.openFile('/data4/guillem/distances/iso384.vort.h5','r')
    data = f.root.vorticity_magnitude.read().astype(np.float32)
    N = data.shape[0]
    data = data/np.sqrt(data.mean()**2 + data.std()**2)
    caseeta = {512: 0.01018, 384: 0.01288, 265: 0.0215}
    eta = caseeta[N]
    print('Kolmogorov scale', eta)

    x = np.linspace(0,2*np.pi,N)/eta
    y = np.linspace(0,2*np.pi,N)/eta
    z = np.linspace(0,2*np.pi,N)/eta

    field = Field(data,x,y,z)

    thresholds = np.logspace(-1,0,5)
    for thres in thresholds:
        print('Computing the pdf for the threshold {}'.format(thres))
        surface = field.extract_complete_surface(thres)
        voxels = surface.oversampled_point_list(field)
        
        sval = data.reshape(N**3)
        side = (2*(np.sign(data > thres).astype(np.int32))-1).reshape(N**3)
        trgt = np.empty((N**3,3),dtype=np.double)
        
        count = 0
        for i,j,k in product(range(N),range(N),range(N)):
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
        contour(10**hist[2][1:],hist[1][1:],np.log(hist[0]))
        semilogx(thresholds[i]*np.ones(hist[2].shape),
                 hist[1],
                 'k--',linewidth=2)
        fig.subplots_adjust(bottom=0.15)
        xlabel(r'$\omega/\omega^\prime$',fontsize=22)
        ylabel(r'$d$',fontsize=22)

    f.close()
