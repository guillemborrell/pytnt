from __future__ import division, print_function
import tables
import numpy as np
from scalar_field import VorticityMagnitudeField
from stats import MiniStats
import pickle
import time

if __name__ == '__main__':
    Results = list()

    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.2000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 400
    NZ = 600

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    print(st.Retau(NX0+OFFSET+NX//2))

    field.scale_outer()
    hists = list()

    thresholds = np.logspace(-4,-3,10)
    hists.append(thresholds)
    for thres in thresholds:
        print(thres)
        clk = time.clock()
        hists.append(field.ball_distance_histogram(thres,200,3000000,150))
        print("distance computations took",time.clock()-clk,"seconds")
        
    f.close()
    st.close()

    #######
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.7000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 400
    NZ = 600

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    print(st.Retau(NX0+OFFSET+NX//2))
    print('This should not be a vector')

    field.scale_outer()
    hists = list()

    thresholds = np.logspace(-4,-3,10)
    hists.append(thresholds)
    for thres in thresholds:
        print(thres)
        clk = time.clock()
        hists.append(field.ball_distance_histogram(thres,200,3000000,150))
        print("distance computations took",time.clock()-clk,"seconds")
        
    f.close()
    st.close()

    #######
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.14000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 400
    NZ = 600

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    print(st.Retau(NX0+OFFSET+NX//2))
    print('This should not be a vector')

    field.scale_outer()
    hists = list()

    thresholds = np.logspace(-4,-3,10)
    hists.append(thresholds)
    for thres in thresholds:
        print(thres)
        clk = time.clock()
        hists.append(field.ball_distance_histogram(thres,200,3000000,150))
        print("distance computations took",time.clock()-clk,"seconds")
        
    f.close()
    st.close()

    Results.append(hists)
        
    resfile = open('/data4/guillem/distances/histogram_ball_outer.1000.dat','w')
    pickle.dump(Results,resfile)
    resfile.close()

