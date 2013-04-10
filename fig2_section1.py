from __future__ import print_function,division
from scalar_field import VorticityComponentField, VorticityMagnitudeField
from stats import MiniStats
import numpy as np
import tables
import pickle
import pylab

if __name__ == '__main__':
    pylab.clf()
    FILENAME = '/data4/guillem/distances/intermittency_characteristics.pickle.dat'
    f = tables.openFile('/data4/guillem/distances/tbl2-HR-Prod.212.real.2000.h5')
    st = MiniStats('/data4/guillem/distances/tbl2-059-271.st.h5',rough=False)
    with open(FILENAME) as resfile:
        Results = pickle.load(resfile)

    NX0 = f.root.NX0.read()
    OFFSET = 50
    NX = 600
    NY = 400
    NZ = 3500
    NDIST = 200

    field = VorticityMagnitudeField(
        f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],
        st,NX0+OFFSET)
    field.scale_outer()
    pdf = np.zeros((NY-1,NDIST),np.double)
    bins = np.linspace(np.log10(field.data.min()),
                       np.log10(field.data.max()),
                       NDIST+1)
    avg = np.zeros((NY-1,),np.double)
    dx = np.diff(bins)

    for j in range(1,NY):
        h,x = np.histogram(np.log10(field.data[:,j,:]),
                           bins = bins)
        x = x[1:]
        norm = np.trapz(10**x*h,x)
        pdf[j-1,:] = h.astype(np.double)/norm*10**x
        avg[j-1] = field.data[:,j,:].mean()
        print(pdf[j-1,:])

    pylab.contour(10**x,field.ydelta[1:],np.log(pdf),12, linewidths=2)
    pylab.plot(field.ydelta**(-0.5),field.ydelta,'k--',linewidth=2)
    pylab.plot(avg,field.ydelta[1:],'k-',linewidth=2)

    field = VorticityComponentField(
        np.abs(f.root.w_z[OFFSET:OFFSET+NX,:NY,:NZ]),
        st,NX0+OFFSET)
    field.scale_outer()
    avg = np.zeros((NY-1,),np.double)

    for j in range(1,NY):
        avg[j-1] = field.data[:,j,:].mean()

    pylab.plot(avg,field.ydelta[1:],'k-',linewidth=2)

    field = VorticityComponentField(
        np.abs(f.root.w_x[OFFSET:OFFSET+NX,:NY,:NZ]),
        st,NX0+OFFSET)
    field.scale_outer()
    avg = np.zeros((NY-1,),np.double)

    for j in range(1,NY):
        avg[j-1] = field.data[:,j,:].mean()

    pylab.plot(avg,field.ydelta[1:],'k-',linewidth=2)


    ax = pylab.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    pylab.ylim([0.02,2.5])
    pylab.xlim([1E-4,1E2])
    pylab.fill_between(x[42:56], 0.02,2.5, facecolor='blue', alpha=0.3)
    
    pylab.xlabel(r'$\omega^*$',fontsize=22)
    pylab.ylabel(r'$y/\delta_{99}$', fontsize=22)
    pylab.savefig('ensprofile.svg')
    pylab.show()
    
