from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    OFFSET = 50
    NX = 600
    NY = 500
    NZ = 2000

    pylab.close("all")
    stages = ['2000','7000','14000']
    threslist =  np.logspace(-2,-0.3,10)
    for stage in stages:
        print('Stage {}...'.format(stage))
        f = tables.openFile(
            '/data4/guillem/distances/tbl2-HR-Prod.212.real.{}.h5'.format(stage))
        st = MiniStats(
            '/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
        st.load_budgets('/data4/guillem/distances/tbl2-059-271.budget.h5')    
        NX0 = f.root.NX0.read()[0]

        field = VorticityMagnitudeField(
            f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],
            st,NX0+OFFSET)
        # field = VorticityComponentField(np.abs(
        #     f.root.w_z[OFFSET:OFFSET+NX,:NY,:NZ]),
        #     st,NX0+OFFSET)
        field.scale_outer()

        for thres in threslist:
            print(thres)
            ydist, ogrid = field.vertical_distance_profile(thres,RANGE=2.0)
            f = pylab.figure(1)
            pylab.plot(ogrid/field.kolmogorov_length_at_height(0.6),
                       ydist,linewidth=1,label=stage)
            
            f = pylab.figure(2)
            pylab.plot(ogrid/field.taylor_microscale_at_height(0.6),
                       ydist,linewidth=1,label=stage)
    
            f = pylab.figure(3)
            pylab.plot(ogrid/st.delta99(NX0+NX/2),
                     ydist,linewidth=1,label=stage)
    
        pylab.figure(1)
        pylab.xlabel(r'$d/\eta$',fontsize=22)
        pylab.ylabel(r'$\omega^*$',fontsize=22)
        f.subplots_adjust(bottom=0.15)
        pylab.savefig('vertical_distance_zoom.svg')

        pylab.figure(2)
        pylab.xlabel(r'$d/\lambda$',fontsize=22)
        pylab.ylabel(r'$\omega^*$',fontsize=22)
        f.subplots_adjust(bottom=0.15)
        pylab.savefig('vertical_distance_zoom.svg')
        
        pylab.figure(3)
        pylab.xlabel(r'$d/\delta_{99}$',fontsize=22)
        pylab.ylabel(r'$\omega^*$',fontsize=22)
        f.subplots_adjust(bottom=0.15)
        pylab.savefig('vertical_distance_zoom.svg')
