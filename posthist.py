from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    pylab.close('all')
    zerooffsets = [-15,5,15,25] # In etas
    stages = ['2000','7000','14000']
    FILENAME = '/data4/guillem/distances/histogram_vertical_cleaned.dat'
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
        NZ = 600
        
        field = VorticityMagnitudeField(
            f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    
        threslist = Results[i][0]

        for j,thres in enumerate(threslist):
            hist = Results[i][j+1][0]
            dist = Results[i][j+1][1]
            logvort = Results[i][j+1][2]

            logvortbin = 0.5*(logvort[1:]+logvort[:-1])
            distbin = 0.5*(dist[1:]+dist[:-1])

            logvort_average = np.array(
                [np.dot(logvortbin,
                        histslice
                        )/histslice.sum() for histslice in hist]
                )

            #Add the first point, that does not appear in the histogram
            distbin[1:] = distbin[:-1]
            distbin[0] = distbin[1]
            logvortbin[1:] = logvortbin[:-1]
            logvortbin[0] = np.log10(thres)
            
            pylab.figure(1)
            pylab.semilogy(distbin[20:]/field.kolmogorov_length_at_height(),
                       10**logvort_average[20:])
            
            pylab.figure(2)
            pylab.semilogy(distbin[20:]/field.taylor_microscale_at_height(),
                       10**logvort_average[20:])

            pylab.figure(3)
            pylab.semilogy(distbin[20:]/st.delta99(NX0+NX/2),
                       10**logvort_average[20:])

            if i == 2 and j == 0:
                pylab.figure(991)
                pylab.contour(field.xr, field.yr, field.data[0,:,:],[thres],linewidths=2)

                for zerooffset in zerooffsets:
                    pastzero = np.where(distbin/field.kolmogorov_length_at_height()>zerooffset)[0][0]
                    print("The point is {}".format(pastzero))
    
                    f = pylab.figure(10)
                    pylab.contour(10**logvortbin,distbin/field.kolmogorov_length_at_height(),
                                  np.log10(hist),linewidths=2)
                    pylab.ylabel(r'$d/\eta$',fontsize=22)
                    pylab.xlabel(r'$\omega^*$',fontsize=22)
                    pylab.plot(thres*np.ones(logvortbin.shape),
                               distbin/field.kolmogorov_length_at_height(),'k--',linewidth=2)
                    pylab.plot(10**logvortbin,np.zeros(distbin.shape),
                               'k--',linewidth=2)
                    pylab.plot(10**logvortbin,
                               distbin[pastzero]*np.ones(distbin.shape)/field.kolmogorov_length_at_height(),
                               'r-',linewidth=2)
                    ax = pylab.gca()
                    ax.set_xscale('log')
                    f.subplots_adjust(bottom=0.15)
    
                    f = pylab.figure(11)
                    pylab.loglog(10**logvortbin,
                                 hist[pastzero,:]/np.trapz(
                            hist[pastzero,:],10**logvortbin),linewidth=2)
                    pylab.ylabel(r'$P(\omega^*)$',fontsize=22)
                    pylab.xlabel(r'$\omega^*$',fontsize=22)
                    f.subplots_adjust(bottom=0.15)
                    

            if i == 2 and j == 9:
                for zerooffset in zerooffsets:
                    pastzero = np.where(
                        distbin/field.kolmogorov_length_at_height()>zerooffset)[0][0]
                    print("The point is {}".format(pastzero))
    
                    f = pylab.figure(20)                
                    pylab.contour(10**logvortbin,distbin/field.kolmogorov_length_at_height(),
                                  np.log10(hist),linewidths=2)
                    pylab.ylabel(r'$d/\eta$',fontsize=22)
                    pylab.xlabel(r'$\omega^*$',fontsize=22)
                    pylab.plot(thres*np.ones(logvortbin.shape),
                               distbin/field.kolmogorov_length_at_height(),'k--',linewidth=2)
                    pylab.plot(10**logvortbin,np.zeros(distbin.shape),
                               'k--',linewidth=2)
                    pylab.plot(10**logvortbin,
                               distbin[pastzero]*np.ones(distbin.shape)/field.kolmogorov_length_at_height(),
                               'r-',linewidth=2)
                    ax = pylab.gca()
                    ax.set_xscale('log')
                    f.subplots_adjust(bottom=0.15)
    
                    f = pylab.figure(21)
                    pylab.loglog(10**logvortbin,
                                 hist[pastzero,:]/np.trapz(
                            hist[pastzero,:],10**logvortbin),linewidth=2)
                    pylab.ylabel(r'$hist(\omega^*)$',fontsize=22)
                    pylab.xlabel(r'$\omega^*$',fontsize=22)
                    f.subplots_adjust(bottom=0.15)
    

    f = pylab.figure(1)
    xlabel(r'$d/\eta$',fontsize=22)
    ylabel(r'$\omega^*$',fontsize=22)
    f.subplots_adjust(bottom=0.15)
    pylab.xlim([-14,300])
    pylab.ylim([0.0005,10])

    f = pylab.figure(2)
    xlabel(r'$d/\lambda$',fontsize=22)
    ylabel(r'$\omega^*$',fontsize=22)
    f.subplots_adjust(bottom=0.15)
    pylab.xlim([-2.5,20])
    pylab.ylim([0.0005,10])
    
    f = pylab.figure(3)
    xlabel(r'$d/\delta_{99}$',fontsize=22)
    ylabel(r'$\omega^*$',fontsize=22)
    f.subplots_adjust(bottom=0.15)
    pylab.xlim([-0.1,1])
    pylab.ylim([0.0005,10])
