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
    
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.2000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
    st.read()

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = np.where(st.y > 2*st.delta99(NX0+NX/2))[0][0]
    NZ = 600

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)

    field.scale_outer()
    hists = list()

    thresholds = np.logspace(-2,-0.3,10)
    hists.append(thresholds)
    for thres in thresholds:
        print(thres)
        hists.append(field.vertical_distance_histogram3d(thres,200).serialize())
        
    f.close()
    st.close()

    Results.append(hists)

    #######
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.7000.h5')
    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = np.where(st.y > 2*st.delta99(NX0+NX/2))[0][0]
    NZ = 600

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)

    field.scale_outer()
    hists = list()

    thresholds = np.logspace(-2,-0.3,10)
    hists.append(thresholds)
    for thres in thresholds:
        hists.append(field.vertical_distance_histogram3d(thres,200).serialize())
        
    f.close()
    st.close()

    Results.append(hists)

    #######
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.14000.h5')
    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = np.where(st.y > 2*st.delta99(NX0+NX/2))[0][0]
    NZ = 600

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)

    field.scale_outer()
    hists = list()

    thresholds = np.logspace(-2,-0.3,10)
    hists.append(thresholds)
    for thres in thresholds:
        hists.append(field.vertical_distance_histogram3d(thres,200).serialize())
        
    f.close()
    st.close()

    Results.append(hists)
        
    resfile = open('/data4/guillem/distances/histogram_vertical_3d.dat','w')
    pickle.dump(Results,resfile)
    resfile.close()

