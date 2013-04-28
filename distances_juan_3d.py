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
        os.remove('./distances_ball_3d.log')
    except OSError:
        pass

    logging.basicConfig(filename='distances.log',
                        level=logging.INFO)
    
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
    st.read()
    FILENAME = '/data4/guillem/distances/histogram_ball_outer_big.dat'
    
    with open(FILENAME) as resfile:
        Histograms = pickle.load(resfile)

    OFFSET = 50
    NX = 600
    NZ = 600

    Results = list()
    thresholds = np.logspace(-2,-0.3,10)

    for i,stage in enumerate(['2000','7000','14000']):
        print(stage)
        hists = list()
        hists.append(thresholds)
        f = tables.openFile(
            '/data4/guillem/distances/tbl2-HR-Prod.212.real.{}.h5'.format(stage))
        NX0 = f.root.NX0.read()
        NY = np.where(st.y > 2*st.delta99(NX0+NX/2))[0][0]
        field = VorticityMagnitudeField(
            f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
        field.scale_outer()

        for j, thres in enumerate(thresholds):
            print(thres)
            dist = Histograms[i][j+1][1]
            logvort = Histograms[i][j+1][2]
            #I still need the histogram for distance to the wall
            height = field.yr[::3]
            hists.append(field.ball_distance_histogram3d(
                    thres,(logvort,dist,height),3000000,150)
                         )
        
        f.close()
        Results.append(hists)

        
    resfile = open('/data4/guillem/distances/histogram_ball_outer_3d.dat','w')
    pickle.dump(Results,resfile)
    st.close()
    resfile.close()

