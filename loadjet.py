import numpy as np

def loadjet(filename,NX=1152,NY=1536,NZ=768):
    fo = open(filename,'r')
    mesh = np.fromfile(fo,dtype=np.float32,count=NX*NY)
    x = mesh[:NX]
    y = mesh[NX:NX+NY]
    z = mesh[NX+NY:NX+NY+NZ]

    return (x,y,z,np.fromfile(fo,dtype=np.float32,count=NX*NY*NZ).reshape(NZ,NY,NX))
