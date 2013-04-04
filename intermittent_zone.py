from __future__ import division, print_function
import tables
import numpy as np
from scalar_field import VorticityMagnitudeField
from stats import MiniStats
import pickle

if __name__ == '__main__':
    Results = list()

    ## Re tau 650 is from Sergio's boundary layer
    f = tables.openFile('/data4/guillem/distances/BL2000.000.real.h5')
    st = MiniStats('/data4/guillem/distances/tbl.2000.141-996.st.h5',rough=False)
    NX0 = 5000
    NX = 600
    NY = 300
    NZ = 1000

    Retau650 = list()
    field = VorticityMagnitudeField(f.root.enstrophy[NX0:NX0+NX,:NY,:NZ],st,NX0)
    print(st.Retau()[NX0])
    field.scale_outer()
    thresholds = np.logspace(-4,-3,10)
    Retau650.append(thresholds)
    Retau650.append(field.ydelta)
    for thres in thresholds:
        print(thres)
        e = field.label_gt_largest(thres)
        s = field.extract_largest_surface(thres)
        Retau650.append((
                field.intermittency_profile(thres),
                e.genus(),
                e.box_counting_exponent(),
                s.genus(),
                s.box_counting_exponent()))

    Results.append(Retau650)

    f.close()
    st.close()

    #### Another Reynolds number

    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.2000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 300
    NZ = 3500

    Retau1000 = list()
    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    print(st.Retau(NX0+OFFSET+NX//2))
    field.scale_outer()
    thresholds = np.logspace(-4,-3,10)
    Retau1000.append(thresholds)
    Retau1000.append(field.ydelta)
    for thres in thresholds:
        print(thres)
        e = field.label_gt_largest(thres)
        s = field.extract_largest_surface(thres)
        Retau1000.append((
                field.intermittency_profile(thres),
                e.genus(),
                e.box_counting_exponent(),
                s.genus(),
                s.box_counting_exponent()))


    Results.append(Retau1000)

    f.close()
    st.close()


    #### Another Reynolds number

    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.7000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 300
    NZ = 3500

    Retau1500 = list()
    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    print(st.Retau(NX0+OFFSET+NX//2))
    field.scale_outer()
    thresholds = np.logspace(-4,-3,10)
    Retau1500.append(thresholds)
    Retau1500.append(field.ydelta)
    for thres in thresholds:
        print(thres)
        e = field.label_gt_largest(thres)
        s = field.extract_largest_surface(thres)
        Retau1500.append((
                field.intermittency_profile(thres),
                e.genus(),
                e.box_counting_exponent(),
                s.genus(),
                s.box_counting_exponent()))


    Results.append(Retau1500)

    f.close()
    st.close()


    #### Another Reynolds number

    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.14000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 300
    NZ = 3500

    Retau2000 = list()
    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    print(st.Retau(NX0+OFFSET+NX//2))
    field.scale_outer()
    thresholds = np.logspace(-4,-3,10)
    Retau2000.append(thresholds)
    Retau2000.append(field.ydelta)
    for thres in thresholds:
        print(thres)
        e = field.label_gt_largest(thres)
        s = field.extract_largest_surface(thres)
        Retau2000.append((
                field.intermittency_profile(thres),
                e.genus(),
                e.box_counting_exponent(),
                s.genus(),
                s.box_counting_exponent()))


    Results.append(Retau2000)

    f.close()
    st.close()


    resfile = open('/data4/guillem/distances/intermittency_characteristics.pickle.dat','w')
    pickle.dump(Results,resfile)
    resfile.close()
