from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    stages = ['2000','7000','14000']
    for stage in stages:
        print('Stage {}...'.format(stage))
        f = tables.openFile(
            '/data4/guillem/distances/tbl2-HR-Prod.212.real.{}.h5'.format(stage))
        st = MiniStats(
            '/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
        st.load_budgets('/data4/guillem/distances/tbl2-059-271.budget.h5')
        
        NX0 = f.root.NX0.read()[0]
        OFFSET = 50
        NX = 800
        NY = 500
        NZ = 4000

        field = VorticityMagnitudeField(
            f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],
            st,NX0+OFFSET)
        field.scale_outer()
        
        ogrid, ydist = field.vertical_distance_profile(0.1,RANGE=4)
        f = figure(1)
        plot(ogrid/field.kolmogorov_length_at_height(0.6),ydist,linewidth=2,label=stage)
        xlabel(r'$d/\eta$',fontsize=22)
        ylabel(r'$\omega^*$',fontsize=22)
        f.subplots_adjust(bottom=0.15)
        savefig('vertical_distance_eta.svg')

        f = figure(2)
        plot(ogrid/field.taylor_microscale_at_height(0.6),ydist,linewidth=2,label=stage)
        xlabel(r'$d/\lambda$',fontsize=22)
        ylabel(r'$\omega^*$',fontsize=22)
        f.subplots_adjust(bottom=0.15)
        savefig('vertical_distance_lambda.svg')


        f = figure(3)
        plot(ogrid/st.delta99(NX0+NX/2),ydist,linewidth=2,label=stage)
        xlabel(r'$d/\delta_{99}$',fontsize=22)
        ylabel(r'$\omega^*$',fontsize=22)
        f.subplots_adjust(bottom=0.15)
        savefig('vertical_distance_delta.svg')

