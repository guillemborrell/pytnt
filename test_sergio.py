from __future__ import division, print_function
import tables
import numpy as np
from scalar_field import VorticityMagnitudeField
from stats import MiniStats
import pickle

if __name__ == '__main__':

    f = tables.openFile('/data4/guillem/distances/BL2000.000.real.h5')
    st = MiniStats('/data4/guillem/distances/tbl.2000.141-996.st.h5',rough=False)

    NX0 = 3000
    NX = 600
    NY = 300
    NZ = 1000

    field = VorticityMagnitudeField(f.root.enstrophy[NX0:NX0+NX,:NY,:NZ],st,NX0)
    print(st.Retau()[NX0])

    field.scale_outer()
    print(field.data.min(), field.data.max())
    hists = list()

    thresholds = np.logspace(-5,-3,10)
    hists.append(thresholds)
    for thres in thresholds:
        hists.append(field.ball_distance_histogram(thres,200,20000000,30))
        
    resfile = open('/data4/guillem/distances/histogram_ball_outer.560.dat','w')
    pickle.dump(hists,resfile)
    resfile.close()

