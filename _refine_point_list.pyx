#cython: boundscheck=False
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef long argmin(double *vect, long N):
    """
    Dummy comment
    """
    cdef int i
    cdef long arg

    arg = 0
    for i in range(N):
        if vect[i] < vect[arg]:
            arg = i
    
    return arg


cpdef _refine_point_list(long N,
                         np.ndarray[long, ndim = 1] voxx,
                         np.ndarray[long, ndim = 1] voxy,
                         np.ndarray[long, ndim = 1] voxz,
                         np.ndarray[double, ndim = 1] xr,
                         np.ndarray[double, ndim = 1] yr,
                         np.ndarray[double, ndim = 1] zr,
                         np.ndarray[float, ndim = 3] data,
                         np.ndarray[double, ndim = 2] pnts,
                         np.ndarray[double, ndim = 2] guess,
                         double thres,
                         long NGUESS):
    """
    Extension for fast refining.
    """
    cdef double dx, dy, dz
    cdef double c00, c01, c10, c11
    cdef double c0, c1
    
    cdef long vx, vy, vz
    cdef long k,cursor

    cdef double *residue = <double *>malloc(NGUESS * sizeof(double))
    cdef double *optim = <double *>malloc(3 * sizeof(double))
    cdef double *vertices = <double *>malloc(8 * sizeof(double))
    for cursor in range(N):
        vx = voxx[cursor]
        vy = voxy[cursor]
        vz = voxz[cursor]
        
        dx = xr[vx+1] - xr[vx]
        dy = yr[vy+1] - yr[vy]
        dz = zr[vz+1] - zr[vz]
        
        for k in range(8):
            vertices[k] = data[vx+k/4, vy+((k/2)%2), vz+(k%2)] - thres
    
        for k in range(NGUESS):
            c00 = vertices[0]*(1-guess[0, k]) + vertices[4]*guess[0, k]
            c01 = vertices[1]*(1-guess[0, k]) + vertices[5]*guess[0, k]
            c10 = vertices[2]*(1-guess[0, k]) + vertices[6]*guess[0, k]
            c11 = vertices[3]*(1-guess[0, k]) + vertices[7]*guess[0, k]
            
            c0 = c00*(1-guess[1, k]) + c01*guess[1, k]
            c1 = c10*(1-guess[1, k]) + c11*guess[1, k]
            
            residue[k] = (c0*(1-guess[2, k]) + c1*guess[2, k])**2
    
        minidx = argmin(residue,NGUESS)
        optim[0] = guess[0, minidx]
        optim[1] = guess[1, minidx]
        optim[2] = guess[2, minidx]
    
        pnts[cursor, 0] = optim[0]*dx + xr[vx]
        pnts[cursor, 1] = optim[1]*dy + yr[vy]
        pnts[cursor, 2] = optim[2]*dz + zr[vz]

    free(residue)
    free(optim)
    free(vertices)

    return 0
