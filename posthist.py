from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    stages = ['2000','7000','14000']
    FILENAME = '/data4/guillem/distances/histogram_ball_outer_big.dat'
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
    st.read()
    st.load_budgets('/data4/guillem/distances/tbl2-059-271.budget.h5')

    with open(FILENAME) as resfile:
        Results = pickle.load(resfile)
        
    for i,stage in enumerate(stages):
        f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.{}.h5'.format(stage))
        NX0 = f.root.NX0.read()[0]
        OFFSET = 50
        NX = 600
        NY = np.where(st.y > 2*st.delta99(NX0+NX/2))[0][0]
        NZ = 10
        
        field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    
        threslist = Results[i][0]
        #pylab.figure(i)
            

        for j,thres in enumerate(threslist):
            hist = Results[i][j+1][0]
            dist = Results[i][j+1][1]
            logvort = Results[i][j+1][2]

            logvortbin = 0.5*(logvort[1:]+logvort[:-1])
            distbin = 0.5*(dist[1:]+dist[:-1])

            limit = np.where(logvortbin > np.log10(thres))[0][0]
            logvort_average = np.array(
                [np.dot(logvortbin[limit:],
                        histslice[limit:]
                        )/histslice[limit:].sum() for histslice in hist]
                )

            print(limit)
            dist_average = np.array(
                [np.dot(distbin,
                        histslice
                        )/histslice.sum() for histslice in hist.T]
                )

            #Add the first point, that does not appear in the histogram
            distbin[1:] = distbin[:-1]
            distbin[0] = 0.0
            logvortbin[1:] = logvortbin[:-1]
            logvortbin[0] = np.log10(thres)
            
            pylab.figure(1)
            pylab.plot(distbin/field.kolmogorov_length_at_height(),logvort_average)
            
            pylab.figure(2)
            pylab.plot(distbin/field.taylor_microscale_at_height(),logvort_average)

            pylab.figure(3)
            pylab.plot(distbin/st.delta99(NX0+NX/2),logvort_average)

            if i == 2 and j == 3:
                f = pylab.figure(10)
                pylab.contour(distbin,logvortbin,np.log(hist),linewidths=2)
                pylab.xlabel(r'$d$',fontsize=22)
                pylab.ylabel(r'$\omega^* dB$',fontsize=22)
                f.subplots_adjust(bottom=0.15)
                pylab.savefig('histogram1.svg')

            if i == 2 and j == 7:
                f = pylab.figure(11)
                pylab.contour(distbin,logvortbin,np.log(hist),linewidths=2)
                pylab.xlabel(r'$d$',fontsize=22)
                pylab.ylabel(r'$\omega^* dB$',fontsize=22)
                f.subplots_adjust(bottom=0.15)
                pylab.savefig('histogram2.svg')


    f = pylab.figure(1)
    xlabel(r'$d/\eta$',fontsize=22)
    ylabel(r'$\omega^*$',fontsize=22)
    f.subplots_adjust(bottom=0.15)
    pylab.savefig('ball_distance_eta.svg')

    f = pylab.figure(2)
    xlabel(r'$d/\lambda$',fontsize=22)
    ylabel(r'$\omega^*$',fontsize=22)
    f.subplots_adjust(bottom=0.15)
    pylab.savefig('ball_distance_lambda.svg')
    
    f = pylab.figure(3)
    xlabel(r'$d/\delta_{99}$',fontsize=22)
    ylabel(r'$\omega^*$',fontsize=22)
    f.subplots_adjust(bottom=0.15)
    pylab.savefig('ball_distance_delta.svg')
