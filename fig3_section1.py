from __future__ import print_function,division
import pickle
import numpy as np
from scipy.interpolate import UnivariateSpline
import pylab

if __name__ == '__main__':
    FILENAME = '/data4/guillem/distances/intermittency_characteristics_outer.pickle.dat' 
    with open(FILENAME) as f:
        Results = pickle.load(f)

    Results_a = np.array(Results)

    for stage in range(4):
        #lkc = np.zeros((len(threslist),),dtype=np.double)
        box_counting_vol = np.array([r[2] for r in Results_a[stage,2:]])
        box_counting_sur = np.array([r[4] for r in Results_a[stage,2:]])
    
        ####
        ydelta = Results_a[stage,1]
        threslist = Results_a[stage,0]
        ####
    
        #Intermittency profile

        Dv = np.empty((len(threslist),9),dtype=np.double)
    
        epsilon = 2**np.linspace(0,8,9)
        for ithres,thres in enumerate(threslist):
            sp = UnivariateSpline(np.log10(epsilon),
                                  np.log10(box_counting_vol[ithres,:9]),
                                  k=5)
            for j in range(len(epsilon)):
                Dv[ithres,j] = sp.derivatives(np.log10(epsilon[j]))[1]
    
        Ds = np.empty((len(threslist),9),dtype=np.double)
        epsilon = 2**np.linspace(0,8,9)
        for ithres,thres in enumerate(threslist):
            sp = UnivariateSpline(np.log10(epsilon),
                                  np.log10(box_counting_sur[ithres,:9]),
                                  k=5)
            for j in range(len(epsilon)):
                Ds[ithres,j] = sp.derivatives(np.log10(epsilon[j]))[1]
    
        gamma = np.array([r[0] for r in Results_a[stage,2:]])
        genusv = np.array([r[1] for r in Results_a[stage,2:]])
        genuss = np.array([r[3] for r in Results_a[stage,2:]])
        
        pylab.figure(1)
        pylab.plot(ydelta,gamma.T)
        pylab.xlim([0,1.5])
        
        pylab.figure(2)
        pylab.semilogx(threslist,Dv[:,:5].mean(axis=1))
        

    pylab.show()
