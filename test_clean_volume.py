from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    FILENAME = '/data4/guillem/distances/intermittency_characteristics.pickle.dat'
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.2000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
    with open(FILENAME) as resfile:
        Results = pickle.load(resfile)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 100
    NY = 400
    NZ = 500

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    field.scale_outer()

    e = field.label_gt_largest(0.1)
    imshow(e.mask()[:,400,:],origin='lower')
