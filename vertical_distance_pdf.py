from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    stages = ['2000','7000','14000']
    threslist =  np.logspace(-2,-0.3,10)
    for s,stage in enumerate(stages):
        print('Stage {}...'.format(stage))
        f = tables.openFile(
            '/data4/guillem/distances/tbl2-HR-Prod.212.real.{}.h5'.format(stage))
        st = MiniStats(
            '/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
        st.load_budgets('/data4/guillem/distances/tbl2-059-271.budget.h5')
        
        NX0 = f.root.NX0.read()[0]
        OFFSET = 50
        NX = 600
        NY = 500
        NZ = 600

        field = VorticityMagnitudeField(
            f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],
            st,NX0+OFFSET)
        # field = VorticityComponentField(np.abs(
        #     f.root.w_z[OFFSET:OFFSET+NX,:NY,:NZ]),
        #     st,NX0+OFFSET)
        field.scale_outer()

        for thres in threslist[:1]:
            hmap = field.interface_height_map(thres)
            f = figure(s)
            pylab.imshow(hmap)
    
