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

    field = VorticityMagnitudeField(f.root.enstrophy[OFFSET:OFFSET+NX,:NY,:NZ],st,NX0+OFFSET)
    field.scale_outer()
    h,x = np.histogram(field.data[:,1:,:],
                       bins=np.logspace(np.log10(field.data.min()),
                                        np.log10(field.data.max()),
                                        100))
    dx = np.diff(x)
    x = x[1:]
    norm = np.trapz(x*h/dx,x)

    fig = pylab.figure(1)
    pylab.clf()
    pylab.semilogx(x,h.astype(np.double)/(norm*dx)*x,'b-',
                   linewidth=2,label=r'$|\omega|^*$')

    fig2 = pylab.figure(2)
    pylab.clf()
    pylab.loglog(x,h.astype(np.double)/(norm*dx),'b-',
                 linewidth=2,label=r'$|\omega|^*$')
    
    field = VorticityComponentField(np.abs(f.root.w_z[OFFSET:OFFSET+NX,:NY,:NZ]),
                                    st,NX0+OFFSET)
    field.scale_outer()
    h,x = np.histogram(field.data[:,1:,:],
                       bins=np.logspace(np.log10(field.data.min()),
                                        np.log10(field.data.max()),
                                        100))
    dx = np.diff(x)
    x = x[1:]
    norm = np.trapz(x*h/dx,x)

    pylab.figure(1)
    pylab.semilogx(x,h.astype(np.double)/(norm*dx)*x,'k--',
                   linewidth=2,label=r'$\omega_z^*$')
    pylab.figure(2)
    pylab.loglog(x,h.astype(np.double)/(norm*dx),'k--',
                 linewidth=2,label=r'$\omega_z^*$')


    field = VorticityComponentField(np.abs(f.root.w_x[OFFSET:OFFSET+NX,:NY,:NZ]),
                                    st,NX0+OFFSET)
    field.scale_outer()
    h,x = np.histogram(field.data[:,1:,:],
                       bins=np.logspace(np.log10(field.data.min()),
                                        np.log10(field.data.max()),
                                        100))

    dx = np.diff(x)
    x = x[1:]
    norm = np.trapz(x*h/dx,x)
    pylab.figure(1)
    pylab.semilogx(x,h.astype(np.double)/(norm*dx)*x,'r-.',
                   linewidth=2,label=r'$\omega_x^*$')
    pylab.figure(2)
    pylab.loglog(x,h.astype(np.double)/(norm*dx),'r-.',
                 linewidth=2,label=r'$\omega_x^*$')

    pylab.figure(1)
    pylab.xlim([1E-4,50])
    #pylab.ylim([1E-4,1E4])
    pylab.legend(loc='best')
    pylab.xlabel(r'$\omega^*$',fontsize=22)
    pylab.ylabel(r'$\omega^*P(\omega^*)$',fontsize=22)
    fig.subplots_adjust(bottom=0.15)
    pylab.savefig('fig1sec1.svg')

    pylab.figure(2)
    pylab.xlim([1E-4,50])
    pylab.ylim([1E-5,1E3])
    pylab.legend(loc='best')
    pylab.xlabel(r'$\omega^*$',fontsize=22)
    pylab.ylabel(r'$P(\omega^*)$',fontsize=22)
    fig2.subplots_adjust(bottom=0.15)
    pylab.savefig('fig1bissec1.svg')


    pylab.show()
