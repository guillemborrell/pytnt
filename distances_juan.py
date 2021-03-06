from __future__ import division,print_function
import tables
import numpy as np
from scalar_field import VorticityMagnitudeField
from stats import MiniStats
import pickle
import logging
import os

if __name__ == '__main__':
    ## Remove previous logging file
    try:
        os.remove('./distances.log')
    except OSError:
        pass

    logging.basicConfig(filename='distances.log',
                        level=logging.INFO)
    Results = list()   
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
    st.read()
    thresholds = np.logspace(-2,-0.3,10)

    OFFSET = 50
    NX = 600
    NZ = 4000

    for stage in ['2000','7000','14000']:
        print(stage)
        f = tables.openFile(
            '/data4/guillem/distances/tbl2-HR-Prod.212.real.{}.h5'.format(stage))
        NX0 = f.root.NX0.read()
        NY = np.where(st.y > 2*st.delta99(NX0+NX/2))[0][0]
        field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
        
        field.scale_outer()
        hists = list()

        hists.append(thresholds)
        for thres in thresholds:
            print(thres)
            hists.append(field.ball_distance_weighted_histogram(
                    thres,500,50000000,150))
        
        f.close()
        Results.append(hists)

    st.close()

    Results.append(hists)
        
    resfile = open('/data4/guillem/distances/histogram_ball_weighted.dat','w')
    pickle.dump(Results,resfile)
    resfile.close()

