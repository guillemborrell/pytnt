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
        
        if stage==1:
            pylab.clf()
            f = pylab.figure(1)
            ax = f.add_subplot(111)
            ax.plot(ydelta,gamma[[4,8,12,16],:].T,linewidth=2)
            pylab.xlim([0,1.5])
            ax.set_ylabel(r'$\gamma$',fontsize=22)
            ax.set_xlabel(r'$y/\delta_{99}$',fontsize=22)
            ax.arrow( 1.1, 0.8, -0.25, -0.2,
                      fc="k", ec="k",
                      head_width=0.02,
                      head_length=0.1)
            ax.text(1.1, 0.7, r'$|\omega^*|$',fontsize=22)
            f.subplots_adjust(bottom=0.15)
            pylab.savefig('intermittency.svg')

        
            f = pylab.figure(2)
            pylab.clf()
            ax1 = f.add_subplot(111)
            ax1.semilogx(threslist,-Dv[:,:3].mean(axis=1),'b-',linewidth=3)
            ax1.set_ylabel(r'$D$',fontsize=22)
            ax1.set_xlabel(r'$|\omega^*|$',fontsize=22)

            ax2 = ax1.twinx()
            pylab.plot(threslist,genusv/genusv.max(),'k--',linewidth=3)
            ax2.set_ylabel(r'$g/max(g)$',fontsize=22)
            
            pylab.plot([threslist[4],threslist[4]],[0,1])
            pylab.plot([threslist[8],threslist[8]],[0,1])
            pylab.plot([threslist[12],threslist[12]],[0,1])
            pylab.plot([threslist[16],threslist[16]],[0,1])

            pylab.xlim([threslist.min(),threslist.max()])
            f.subplots_adjust(bottom=0.15)
            f.subplots_adjust(right=0.85)
            pylab.savefig('genus_fractal.svg')
        
    pylab.show()
