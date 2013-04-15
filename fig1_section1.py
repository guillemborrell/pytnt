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
    NZ = 3500

    field = VorticityMagnitudeField(
        f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    field.scale_outer()
    h,x = np.histogram(np.log10(field.data[:,1:,:]),bins=100)
    x = x[1:]
    norm = np.trapz(h,x)

    fig = pylab.figure(1)
    pylab.clf()
    pylab.semilogx(10**x,h.astype(np.double)/norm,'b-',
                   linewidth=2,label=r'$|\omega|^*$')

    
    field = VorticityComponentField(
        np.abs(f.root.w_z[OFFSET:OFFSET+NX,:NY,:NZ]),st,NX0+OFFSET)
    field.scale_outer()
    h,x = np.histogram(np.log10(field.data[:,1:,:]),
                       bins=100)
    x = x[1:]
    norm = np.trapz(h,x)
    
    pylab.figure(1)
    pylab.semilogx(10**x,h.astype(np.double)/norm,'k--',
                   linewidth=2,label=r'$\omega_z^*$')


    field = VorticityComponentField(np.abs(f.root.w_x[OFFSET:OFFSET+NX,:NY,:NZ]),
                                    st,NX0+OFFSET)
    field.scale_outer()
    h,x = np.histogram(np.log10(field.data[:,1:,:]),
                       bins=100)

    x = x[1:]
    norm = np.trapz(h,x)
    pylab.figure(1)
    pylab.semilogx(10**x,h.astype(np.double)/norm,'r-.',
                   linewidth=2,label=r'$\omega_x^*$')

    pylab.figure(1)
    pylab.xlim([1E-4,50])
    #pylab.ylim([1E-4,1E4])
    pylab.legend(loc='best')
    pylab.xlabel(r'$\omega^*$',fontsize=22)
    pylab.ylabel(r'$P(\omega^*dB)$',fontsize=22)
    fig.subplots_adjust(bottom=0.15)
    pylab.savefig('fig1sec1.svg')

    pylab.show()
