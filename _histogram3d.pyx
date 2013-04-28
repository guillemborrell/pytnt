cimport numpy as np

cpdef _histogram3d(np.ndarray[float, ndim=2] data,
                   long ndata,
                   np.ndarray[double, ndim=1] binsx,
                   long nbinsx,
                   np.ndarray[double, ndim=1] binsy,
                   long nbinsy,
                   np.ndarray[double, ndim=1] binsz,
                   long nbinsz,
                   np.ndarray[long, ndim=3] hist):
    """
    Compute a 3d histogram. The fast way. It is slightly different
    from the traditional histogram, because it adds two extra bins at
    the limits of the histogram to check if there are values that
    spill the bin distribution.  Then, if the bins have lengths of NX,
    NY and NZ, the array for the histogram should be NZ+1, NY+1 and
    NZ+1. But don't worry, if you screw, probably it will segfault or
    something similar.

    The hist array should be initalized to zero, but not enforced in
    case you want to build the histogram incrementally.
    
    Signature::

        (np.ndarray[float, ndim=2], long, np.ndarray[double, ndim=1],
        long, np.ndarray[double, ndim=1], long, np.ndarray[double, ndim=1],
        long, np.ndarray[long, ndim=3])
    """
    cdef int counter,i,j,k

    for counter in range(ndata):
        i = 0
        j = 0
        k = 0
        if binsx[0] > data[0, counter]:
            i = 0
        elif binsx[nbinsx-1] < data[0, counter]:
            i = nbinsx
        else:
            while binsx[i] < data[0, counter]:
                i += 1

        if binsy[0] > data[1, counter]:
            j = 0
        elif binsy[nbinsy-1] < data[1, counter]:
            j = nbinsy
        else:
            while binsy[j] < data[1, counter]:
                j += 1
            
        if binsz[0] > data[2, counter]:
            k = 0
        elif binsz[nbinsz-1] < data[2, counter]:
            k = nbinsz
        else:
            while binsz[k] < data[2,counter]:
                k += 1

        hist[i,j,k] += 1
