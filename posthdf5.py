try:
    from mpi4py import MPI
    PARALLEL = True
except ImportError:
    print "No support for parallel statistics"
    PARALLEL = False

from itertools import count,product,chain
import numpy as np
import numpy.ma as ma
import numexpr as ne
from scipy import interpolate
from scipy.weave import inline, converters
from time import clock
import tables

#Extension
#from otree_fast import select_lower

gaussian_kernel = np.array([.006,.061,.242,.383,.242,.061,.006],dtype=np.float32)

class MiniStats(object):
    def __init__(self,fname,comm=None,rough=True,parallel=True):
        self.ROUGH = rough
        self.PARALLEL = parallel and PARALLEL
        self.comm = comm
        if self.PARALLEL: 
            self.rank = comm.Get_rank()
        else:
            self.rank = 0

        self.budgets = None
        self.fname = fname
        self.utau = None
        self.Re = None
        self.NX = None
        self.NY = None
        self.NZ = None
        self.x = None
        self.z = None
        self.y = None
        self.yr = None
        self.ua = None
        self.us = None
        self.va = None
        self.vs = None
        self.uv = None
        self.wa = None
        self.ws = None
        self.wa = None
        self.ws = None
        self.utau = None
        self.dx = None
        self.dz = None
        
    def read(self):
        """Only master can do this"""
        if self.rank == 0:
            self.stats = tables.openFile(self.fname,mode='r')

            self.NX = self.stats.root.nx.read()[0]
            self.NY = self.stats.root.ny.read()[0]
            self.NZ = self.stats.root.nz2.read()[0]*2

            self.Re = self.stats.root.Re.read()[0]
            self.x = np.linspace(0,self.stats.root.ax.read()*np.pi,self.NX)
            self.z = np.linspace(0,self.stats.root.az.read()*np.pi*2,self.NZ)
            self.y = self.stats.root.y.read()
            self.yr = self.y[1:-1]
            self.dx = self.stats.root.ax.read()*np.pi
            self.dz = self.stats.root.az.read()*2*np.pi

            self.ua = self.stats.root.ua.read()
            self.us = self.stats.root.us.read()
            self.va = self.stats.root.va.read()
            self.vs = self.stats.root.vs.read()
            self.wa = self.stats.root.wa.read()
            self.ws = self.stats.root.ws.read()
            self.uv = self.stats.root.uv.read()
            self.pa = self.stats.root.pm.read()
            self.ps = self.stats.root.pp.read()

            vortza_w = self.stats.root.vortza.read()[:,0]
            wstr = -1/self.Re*vortza_w

            if self.ROUGH:
                deltatau = np.empty(wstr.shape)
                gshape = 1.2*(-np.tanh((self.yr-0.13602)/0.036)+1)/(2.*18.)
                
                for i in range(len(wstr)):
                    deltatau[i] = np.trapz(self.yr,gshape*self.ua[i,:])

                wstr += np.abs(deltatau)

            self.utau = np.sqrt(wstr)

            self.us = np.sqrt(np.abs(self.us - self.ua**2))
            self.vs = np.sqrt(np.abs(self.vs - self.va**2))
            self.ws = np.sqrt(np.abs(self.ws - self.wa**2))
            self.ps = np.sqrt(np.abs(self.ps - self.pa**2))
            self.uv = np.sqrt(np.abs(self.uv - self.ua*self.va))

        else:
            print "You are not the master, you don't read"


    def broadcast(self):
        """Share the stats with the rest of the nodes"""

        self.NX = self.comm.bcast(self.NX,root=0)
        self.NY = self.comm.bcast(self.NY,root=0)
        self.NZ = self.comm.bcast(self.NZ,root=0)
        self.Re = self.comm.bcast(self.Re,root=0)
        self.x  = self.comm.bcast(self.x ,root=0)
        self.y  = self.comm.bcast(self.y ,root=0)
        self.z  = self.comm.bcast(self.z ,root=0)
        self.yr = self.comm.bcast(self.yr,root=0)
        self.ua = self.comm.bcast(self.ua,root=0)
        self.us = self.comm.bcast(self.us,root=0)
        self.va = self.comm.bcast(self.va,root=0)
        self.vs = self.comm.bcast(self.vs,root=0)
        self.wa = self.comm.bcast(self.wa,root=0)
        self.ws = self.comm.bcast(self.ws,root=0)
        self.uv = self.comm.bcast(self.uv,root=0)
        self.pa = self.comm.bcast(self.pa,root=0)
        self.ps = self.comm.bcast(self.ps,root=0)

        self.utau = self.comm.bcast(self.utau,root=0)
        self.dx = self.comm.bcast(self.dx,root=0)
        self.dz = self.comm.bcast(self.dz,root=0)

    def ua_spline(self):
        return interpolate.RectBivariateSpline(self.x,self.yr,self.ua)

    def close(self):
        if self.rank == 0: self.stats.close()
        else: print "You can't close the file, you're rank {}".format(self.rank)

    def Ue(self):
        return np.mean(self.ua[:,-15:-5],axis=1)

    def theta(self):
        Ue = self.Ue()
        res = np.empty((self.NX,))
        for i in range(self.NX):
            res[i] = np.trapz(
                self.ua[i,:]/Ue[i]*(1-self.ua[i,:]/Ue[i]),
                self.yr)

        return res

    def deltastar(self):
        Ue = self.Ue()
        res = np.empty((self.NX,))
        for i in range(self.NX):
            res[i] = np.trapz((1-self.ua[i,:]/Ue[i]),self.yr)

        return res

    def delta99(self):
        res = np.empty((self.NX,))
        for i in range(self.NX):
            uint = interpolate.interp1d(self.ua[i,:],self.yr)
            res[i] = uint(0.99)
        return res
        

    def Reth(self):
        theta = self.theta()
        Ue = self.Ue()
        return theta*self.Re*Ue

    def Retau(self):
        d99 = self.delta99()
        return d99*self.utau*self.Re

    def ydelta(self,i):
        return self.yr/self.delta99()[i]

    def xdelta(self,i):
        return self.x/self.delta99()[i]

    def zdelta(self,i):
        return self.z/self.delta99()[i]

    def load_budgets(self,budgets_file):
        self.budgets = tables.openFile(budgets_file,mode='r')
        
    def dissipation(self):
        dudx0 = self.budgets.root.dudx0.read()
        dudy0 = self.budgets.root.dudy0.read()
        dvdx0 = self.budgets.root.dvdx0.read()
        dvdy0 = self.budgets.root.dvdy0.read()
        dwdx0 = self.budgets.root.dwdx0.read()
        dwdy0 = self.budgets.root.dwdy0.read()
        
        dispu = self.budgets.root.dispu.read()
        dispv = self.budgets.root.dispv.read()
        dispw = self.budgets.root.dispw.read()
        dispuv = self.budgets.root.dispuv.read()

        #kinematic viscosity and zero mode
        dispu  = (dispu  - (dudx0**2 - dudy0**2))*(2/self.Re)
        dispv  = (dispv  - (dvdx0**2 - dvdy0**2))*(2/self.Re)
        dispw  = (dispw  - (dwdx0**2 - dwdy0**2))*(2/self.Re)
        dispuv = (dispuv - (dudx0*dvdx0 - dudy0*dvdy0))*(2/self.Re)

        #Wall units
        scab = 1/(self.utau**4 * self.Re)
        for i in range(self.NX):
            dispu[i,:] = dispu[i,:]*scab[i]
            dispv[i,:] = dispv[i,:]*scab[i]
            dispw[i,:] = dispw[i,:]*scab[i]
            dispuv[i,:] = dispuv[i,:]*scab[i]

        #Return only ke dissipation atm.
        return 0.5*(dispu+dispv+dispw)

    def production(self):
        dudx0 = self.budgets.root.dudx0.read()
        dudy0 = self.budgets.root.dudy0.read()
        dvdx0 = self.budgets.root.dvdx0.read()
        dvdy0 = self.budgets.root.dvdy0.read()

        produ = -2*self.us*dudx0 -2*self.uv*dudy0
        prodv = -2*self.uv*dvdx0 -2*self.vs*dvdy0
        produv = -self.us*dvdx0 -self.vs*dudy0

        #Wall units
        scab = 1/(self.utau**4 * self.Re)
        for i in range(self.NX):
            produ[i,:] = produ[i,:]*scab[i]
            prodv[i,:] = prodv[i,:]*scab[i]
            produv[i,:] = produv[i,:]*scab[i]

        return 0.5*(produ + prodv)


class PostField(object):
    def __init__(self,fname,comm,rank,size,debug=False):
        self.fname = fname
        self.comm = comm
        self.rank = rank
        self.size = size
        self.debug = debug
        (self.NX,
         self.NY,
         self.NZ,
         self.NPX,
         self.NPZ) = (0,0,0,0,0)
        self.fid = None
        self.data = None
        
    def read_serial(self,start=(0,0,0),dims=(0,0,0)):
        """Setting one dim to 0 means read all the dimension"""

        if self.rank == 0:
            self.fid = tables.openFile(self.fname,'r')

            if dims[2] == 0:
                NZR = self.fid.root.value.shape[2]-start[2]
            else:
                NZR = dims[2]-start[2]

            if dims[1] == 0: 
                NYR = self.fid.root.value.shape[1]-start[1]
            else:
                NYR = dims[1]-start[1]

            if dims[0] == 0: 
                NXR = self.fid.root.value.shape[0]-start[0]
            else:
                NXR = dims[0]-start[0]

            
            self.NPX = NXR/self.size
            self.NPZ = NZR/self.size
            
            self.NX = self.NPX*self.size
            self.NZ = self.NPZ*self.size
            self.NY = NYR
            
            print "-REPORT"
            print "-FILE:",self.fid.root.value.shape
            print "-CHUNK: start, size",start,(NXR,NYR,NZR)
            print "-LOADED:",self.NX,self.NY,self.NZ
            print "   Per node:",self.NPX,self.NPZ
            print "-END REPORT"
        
        self.NX = self.comm.bcast(self.NX,root=0)
        self.NY = self.comm.bcast(self.NY,root=0)
        self.NZ = self.comm.bcast(self.NZ,root=0)
        self.NPX = self.comm.bcast(self.NPX,root=0)
        self.NPZ = self.comm.bcast(self.NPZ,root=0)
        self.data = np.empty((self.NPX,self.NY,self.NZ),dtype=np.float32)

        for (i,pl) in zip(range(self.size-1,0,-1),
                          range(self.NX-self.NPX,0,-self.NPX)):
            if self.rank == 0:
                print "Reading chunk #",i
                self.data = self.fid.root.value[start[0]+pl:pl+self.NPX,
                                                start[1]:self.NY,
                                                start[2]:self.NZ]
                self.comm.Send(self.data,dest=i,tag=self.size+i)

            elif self.rank == i:
                self.comm.Recv(self.data,source=0,tag=self.size+i)
   
        if self.rank == 0:
            self.data = self.fid.root.value[start[0]:self.NPX,
                                            start[1]:self.NY,
                                            start[2]:self.NZ]

        print "INFO: rank {} has shape {}".format(self.rank,
                                                  self.data.shape)


    def fou2phys(self):
        """Go to physical space"""
        shape = self.data.shape
        fourier = np.empty((shape[0],shape[1],shape[2]/2),
                           dtype=np.complex64)
        
        fourier.real[:,:,:] = self.data[:,:,::2]
        fourier.imag[:,:,:] = self.data[:,:,1::2]
        self.data = np.fft.irfft(fourier)[:,:,:]
        self.NZ = self.NZ-2


    def filter_phys_z(self,kernel):
        """Filter in Physical space"""
        for i in range(self.NPX):
            for j in range(self.NY):
                self.data[i,j,:] = np.convolve(
                    self.data[i,j,:],kernel,mode='same')
                

    def filter_phys_xy(self,kernel):
        """Filter in Physical space"""
        for k in range(self.NPZ):
            for i in range(self.NY):
                self.data[k,i,:] = np.convolve(
                    self.data[k,i,:],kernel,mode='same')
                
            for j in range(self.NX):
                self.data[k,:,j] = np.convolve(
                    self.data[k,:,j],kernel,mode='same')
                    
                    
    def resample(self,stride):
        """Resample points. Done in zy"""
        buff = np.empty(((self.NPX+stride-1)/stride,
                         (self.NY+stride-1)/stride,
                         (self.NZ+stride-1)/stride),dtype=np.float32)
        buff[:,:,:] = self.data[::stride,::stride,::stride]
        self.data = buff
        self.NPX = (self.NPX+stride-1)/stride
        self.NPZ = self.NZ/self.size
        self.NX = self.NPX*self.size
        self.NY = (self.NY+stride-1)/stride
        self.NZ = (self.NZ+stride-1)/stride

        
    def dump_serial(self,filename):
        if self.rank == 0:
            dfile = tables.openFile(filename,'w')
            earray = dfile.createEArray(dfile.root,'value',
                                        tables.atom.Float32Atom(),
                                        (0,self.NY,self.NZ))
            earray.append(self.data)

        for i in range(1,self.size):
            if self.rank == 0:
                self.comm.Recv(self.data,source=i,tag=self.size+i)
                earray.append(self.data)

            elif self.rank == i:
                self.comm.Send(self.data,dest=0,tag=self.size+i)

        if self.rank == 0: dfile.close()

    def chzy2xy(self):
        """Changes. It uses a lot of memory"""
        ssbuf = np.empty((self.NZ,self.NPX,self.NY),dtype=np.float32)
        rrbuf = np.empty((self.size,self.NPZ,self.NPX,self.NY),dtype=np.float32)
        

        ssbuf[:,:,:] = np.transpose(self.data,(2,0,1))[:,:,:]
        self.comm.Alltoall([ssbuf, self.NY*self.NPZ*self.NPX, MPI.FLOAT],
                           [rrbuf, self.NY*self.NPZ*self.NPX, MPI.FLOAT])

        del ssbuf
        rbuf = np.empty((self.NPZ,self.NY,self.size,self.NPX),dtype=np.float32)
        rbuf[:,:,:,:] = np.transpose(rrbuf,(1,3,0,2))[:,:,:,:]
        self.data = rbuf.reshape((self.NPZ,self.NY,self.NX))
        
    def chxy2zy(self):
        """Changes. It again uses a lot of memory"""
        ssbuf = np.empty((self.size,self.NPZ,self.NPX,self.NY),dtype=np.float32)
        rrbuf = np.empty((self.NZ,self.NPX,self.NY),dtype=np.float32)
        
        ssbuf[:,:,:,:] = np.transpose(
            self.data.reshape((self.NPZ,self.NY,self.size,self.NPX)),
            (2,0,3,1))[:,:,:,:]
        self.comm.Alltoall([ssbuf, self.NY*self.NPZ*self.NPX, MPI.FLOAT],
                           [rrbuf, self.NY*self.NPZ*self.NPX, MPI.FLOAT])
        
        del ssbuf
        rbuf = np.empty((self.NPX,self.NY,self.NZ),dtype=np.float32)
        rbuf[:,:,:] = np.transpose(rrbuf,(1,2,0))[:,:,:]
        self.data = rbuf
      
    def close(self):
        if self.rank == 0: self.fid.close()
        else: print "You can't close the file, you're rank {}".format(self.rank)


class Geometry(object):
    """
    Classify the field geometrically
    """
    def __init__(self,data,x_grid,y_grid,z_grid):
        self.data = data
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = z_grid

    def __genus(self,domain):
        (NX,NY,NZ) = domain.shape
        
    #Vertices
        vertices = np.zeros((NX+1,NY+1,NZ+1),dtype=np.int8)
        vertarray = np.zeros((NX-1,NY-1,NZ-1,8),dtype=np.int8)
        
    # Fill the vertex array
        vertarray[:,:,:,0] = domain[:-1,:-1,:-1]
        vertarray[:,:,:,1] = domain[:-1,:-1,1:]
        vertarray[:,:,:,2] = domain[:-1,1:,:-1]
        vertarray[:,:,:,3] = domain[:-1,1:,1:]
        vertarray[:,:,:,4] = domain[1:,:-1,:-1]
        vertarray[:,:,:,5] = domain[1:,:-1,1:]
        vertarray[:,:,:,6] = domain[1:,1:,:-1]
        vertarray[:,:,:,7] = domain[1:,1:,1:]
        
        vertices[1:-1,1:-1,1:-1] = np.logical_xor(
            vertarray.all(axis=3),
            vertarray.any(axis=3)).astype(np.int8)
        
    #Sum all the vertices
        nvertices = vertices.sum()
        del vertices
        del vertarray
        
    #Edges
    #Edges x are the edges aligned in the x direction
        edgesx = np.zeros((NX,NY+1,NZ+1),dtype=np.int8)
        edgesxarray = np.zeros((NX,NY-1,NZ-1,4),dtype=np.int8)
        
    ##### EDGES
    #Fill edgesx
        edgesxarray[:,:,:,0] = domain[:,:-1,:-1]
        edgesxarray[:,:,:,1] = domain[:,:-1,1:]
        edgesxarray[:,:,:,2] = domain[:,1:,:-1]
        edgesxarray[:,:,:,3] = domain[:,1:,1:]
        
        edgesx[:,1:-1,1:-1] = np.logical_xor(
            edgesxarray.all(axis=3),
            edgesxarray.any(axis=3)).astype(np.int8)
        
        nedges = edgesx.sum()
        
        del edgesx
        del edgesxarray
        
        edgesy = np.zeros((NX+1,NY,NZ+1),dtype=np.int8)
        edgesyarray = np.zeros((NX-1,NY,NZ-1,4),dtype=np.int8)
        
    #Fill edgesy
        edgesyarray[:,:,:,0] = domain[:-1,:,:-1]
        edgesyarray[:,:,:,1] = domain[:-1,:,1:]
        edgesyarray[:,:,:,2] = domain[1:,:,:-1]
        edgesyarray[:,:,:,3] = domain[1:,:,1:]
        
        edgesy[1:-1,:,1:-1] = np.logical_xor(
            edgesyarray.all(axis=3),
            edgesyarray.any(axis=3)).astype(np.int8)
        
        nedges += edgesy.sum()
        
        del edgesy
        del edgesyarray
        
        edgesz = np.zeros((NX+1,NY+1,NZ),dtype=np.int8)
        edgeszarray = np.zeros((NX-1,NY-1,NZ,4),dtype=np.int8)
        
    #Fill edgesz
        edgeszarray[:,:,:,0] = domain[:-1,:-1,:]
        edgeszarray[:,:,:,1] = domain[:-1,1:,:]
        edgeszarray[:,:,:,2] = domain[1:,:-1,:]
        edgeszarray[:,:,:,3] = domain[1:,1:,:]
        
        edgesz[1:-1,1:-1,:] = np.logical_xor(
            edgeszarray.all(axis=3),
            edgeszarray.any(axis=3)).astype(np.int8)
        
        nedges += edgesz.sum()
        
        del edgesz
        del edgeszarray
        
    #Faces
    #Faces x are the faces normal in the x direction
        facesx = np.zeros((NX+1,NY,NZ),dtype=np.int8)
        facesxarray = np.zeros((NX-1,NY,NZ,2),dtype=np.int8)
        
    #### FACES
    # Fill sidex
        facesxarray[:,:,:,0] = domain[:-1,:,:]
        facesxarray[:,:,:,1] = domain[1:,:,:]
        
        facesx[1:-1,:,:] = np.logical_xor(
            facesxarray.all(axis=3),
            facesxarray.any(axis=3)).astype(np.int8)
        
        nfaces = facesx.sum()
        
        del facesx
        del facesxarray
        
        facesy = np.zeros((NX,NY+1,NZ),dtype=np.int8)
        facesyarray = np.zeros((NX,NY-1,NZ,2),dtype=np.int8)
        
    # Fill sidey
        facesyarray[:,:,:,0] = domain[:,:-1,:]
        facesyarray[:,:,:,1] = domain[:,1:,:]
        
        facesy[:,1:-1,:] = np.logical_xor(
            facesyarray.all(axis=3),
            facesyarray.any(axis=3)).astype(np.int8)
        
        nfaces += facesy.sum()
        
        del facesy
        del facesyarray
        
        facesz = np.zeros((NX,NY,NZ+1),dtype=np.int8)
        faceszarray = np.zeros((NX,NY,NZ-1,2),dtype=np.int8)
        
    # Fill sidez
        faceszarray[:,:,:,0] = domain[:,:,:-1]
        faceszarray[:,:,:,1] = domain[:,:,1:]
        
        facesz[:,:,1:-1] = np.logical_xor(
            faceszarray.all(axis=3),
            faceszarray.any(axis=3)).astype(np.int8)
        
        nfaces += facesz.sum()
        
    # print nvertices,nedges,nfaces
        xi = nvertices-nedges+nfaces
    # print 'Euler Charactersitic', xi
        g = (2-xi)/2
    # print 'Genus', g
        return np.array([nvertices,nedges,nfaces,xi,g],dtype=np.int64)

    def genus_object(self,field):
        """
        Computes the genus tuple of a given object.

        Remember that it must be enclosed in zeros. It can receive the
        """
        return self.__genus(field)

    def genus(self,thres):
        """
        Genus with the data enclosed in a box of zeros. Enclosing it
        with zeros will make you se objects below the threshold as
        individual objects inside the higher-than-threshold body.
        """
        dshape = (self.data.shape[0]+2,
                  self.data.shape[1]+2,
                  self.data.shape[2]+2)
        domain = np.zeros(dshape,dtype=np.int8)
        domain[1:-1,1:-1,1:-1] = (self.data[:,:,:]>thres).astype(np.int8)
        print "Computing genus for %f"%(thres)
        return self.__genus(domain)

    
    def __stencilize_par(self,clusterbed):
        codestring ="""
int i,j,k;
int counter = 0;
int tid;

//#pragma omp parallel private(tid)
//{
//  tid = omp_get_thread_num();
//  printf("Hello World from thread = %d", tid);
//}

#pragma omp parallel for private(i,j,k,tid)
for (i=1;i<nx-1;i++){
  for (j=1;j<ny-1;j++){
    for (k=1;k<nz-1;k++){
      clusterbed(i,j,k,1) = clusterbed(i,j,k-1,0);
      clusterbed(i,j,k,2) = clusterbed(i,j,k+1,0);
      clusterbed(i,j,k,3) = clusterbed(i,j-1,k,0);
      clusterbed(i,j,k,4) = clusterbed(i,j+1,k,0);
      clusterbed(i,j,k,5) = clusterbed(i-1,j,k,0);
      clusterbed(i,j,k,6) = clusterbed(i+1,j,k,0);
    }
  }
}
        """
        nx = clusterbed.shape[0]
        ny = clusterbed.shape[1]
        nz = clusterbed.shape[2]
        variables = "clusterbed nx ny nz".split()
        inline(codestring,
               variables,
               extra_compile_args =['-O3 -fopenmp'],
               compiler='gcc',
               extra_link_args=['-lgomp'],
               headers=['<cmath>','<omp.h>','<cstdio>'],
               type_converters=converters.blitz)
        

    def __stencilize(self,clusterbed):
        clusterbed[1:-1,1:-1,1:-1,1] = clusterbed[1:-1,1:-1,:-2,0]
        clusterbed[1:-1,1:-1,1:-1,2] = clusterbed[1:-1,1:-1,2:,0]
        clusterbed[1:-1,1:-1,1:-1,3] = clusterbed[1:-1,:-2,1:-1,0]
        clusterbed[1:-1,1:-1,1:-1,4] = clusterbed[1:-1,2:,1:-1,0]
        clusterbed[1:-1,1:-1,1:-1,5] = clusterbed[:-2,1:-1,1:-1,0]
        clusterbed[1:-1,1:-1,1:-1,6] = clusterbed[2:,1:-1,1:-1,0]

    def __maximize_par(self,clusterbed,domain):
        codestring ="""                              
using namespace std;
int i,j,k;                               
                                                           
#pragma omp parallel for private(i,j,k)                         
for (i=1;i<nx-1;i++){                                            
  for (j=1;j<ny-1;j++){                                  
    for (k=1;k<nz-1;k++){                                    
      clusterbed(i,j,k,0) = max(                  
                            max(                     
                            max(                        
                            max(                     
                            max(                        
                            max(clusterbed(i,j,k,0),        
                                clusterbed(i,j,k,1)),           
                                clusterbed(i,j,k,2)),         
                                clusterbed(i,j,k,3)),        
                                clusterbed(i,j,k,4)),          
                                clusterbed(i,j,k,5)),        
                                clusterbed(i,j,k,6))*domain(i,j,k); 
    }                                                         
  }                                                           
}                                                      
        """
        nx = clusterbed.shape[0]
        ny = clusterbed.shape[1]
        nz = clusterbed.shape[2]
        variables = "clusterbed domain nx ny nz".split()
        inline(codestring,
               variables,
               extra_compile_args =['-O3 -fopenmp'],
               extra_link_args=['-lgomp'],
               compiler='gcc',
               headers=['<cmath>','<omp.h>'],
               type_converters=converters.blitz)

    def __objects(self,domain):
        #Put the biggest number in the center        
        flipcounter = lambda NN: np.concatenate(
            (np.arange(NN,dtype=np.uint32)[NN/2:],
             np.arange(NN,dtype=np.uint32)[:NN/2]))

        (NX,NY,NZ) = domain.shape
        stencil = (NX,NY,NZ,7)
        NITMAX = np.max([NX,NY,NZ])

        generator = np.zeros(domain.shape,dtype=np.uint32)

        for i in flipcounter(NX):
            for j in flipcounter(NY):
                line = 1+flipcounter(NZ)
                generator[i,j,:] = line+j*NZ+i*NZ*NY

        print "Generator built" #DEBUG 
        clusterbed = np.zeros(stencil,dtype=np.uint32)
        clusterbed[:,:,:,0] = generator*domain

        for i in range(3*NITMAX):
            if i%100 == 0: print 'step: ',i
            # Stop condition that doesn't always work
            # if i == NITMAX/2:
            #     print "start comparing item list at %i"%(i)
            #     itlist = np.unique(clusterbed[:,:,:,0]).flatten()
            # if i%20 == 0 and i>NITMAX/2:
            #     newitlist = np.unique(clusterbed[:,:,:,0]).flatten()
            #     if len(itlist) == len(newitlist):
            #         print "Stop iterating at",i
            #         break
            #     else:
            #         itlist = newitlist
            #         print len(itlist),"not equal yet"

            self.__stencilize_par(clusterbed)
            generator = clusterbed[:,:,:,0]
            #clusterbed[:,:,:,0] = clusterbed.max(axis=3)*domain
            self.__maximize_par(clusterbed,domain)

        return (np.unique(clusterbed[:,:,:,0].reshape(domain.shape)),
                clusterbed[:,:,:,0].reshape(domain.shape))


        return (np.unique(clusterbed[:,:,:,0].reshape(domain.shape)),
                clusterbed[:,:,:,0].reshape(domain.shape))


    def objects(self,thres):
        """
        Returns an object list and the object field given a threshold
        for the data
        """
        dshape = (self.data.shape[0]+2,
                  self.data.shape[1]+2,
                  self.data.shape[2]+2)
        domain = np.zeros(dshape,dtype=np.int8)
        domain[1:-1,1:-1,1:-1] = (self.data[:,:,:]>thres).astype(np.int8)
        return self.__objects(domain)


    def objects_in(self,thres):
        """
        Returns an object list and the object field given a threshold
        for the data.

        INSIDE VERSION
        """
        dshape = (self.data.shape[0]+2,
                  self.data.shape[1]+2,
                  self.data.shape[2]+2)
        domain = np.zeros(dshape,dtype=np.int8)
        domain[1:-1,1:-1,1:-1] = (self.data[:,:,:]<thres).astype(np.int8)
        return self.__objects(domain)

    def __getvolumeobject_par(self,objfield,obj,dvol):
        codestring ="""
int i,j,k;
volume = 0.0;
unsigned int point;
unsigned int object;

object = int(obj);

#pragma omp parallel for reduction(+ : volume) private(i,j,k)
for (i=1;i<nx-1;i++){                                    
  for (j=1;j<ny-1;j++){                                                   
    for (k=1;k<nz-1;k++){
      point = objfield(i,j,k);
      if (point == object){
        volume = volume + dvol(i+1,j+1,k+1);
      }
    }                                                                   
  }                                                                 
}                                                                       
        """
        volume = np.empty((1,),dtype=np.float64)
        nx = objfield.shape[0]
        ny = objfield.shape[1]
        nz = objfield.shape[2]
        variables = "objfield obj dvol volume nx ny nz".split()
        inline(codestring,
               variables,
               extra_compile_args =['-O3 -fopenmp'],
               extra_link_args=['-lgomp'],
               headers=['<cmath>'],
               compiler='gcc',
               type_converters=converters.blitz)

        return volume
    
    def volume_object(self,field,obj):
        """
        Computes the volume of a given object
        """
        dx = self.x_grid[1:] - self.x_grid[:-1]
        dy = self.y_grid[1:] - self.y_grid[:-1]
        dz = self.z_grid[1:] - self.z_grid[:-1]

        dvol = np.kron(np.kron(dx,dy).reshape(self.data.shape[0],
                                              self.data.shape[1]),
                       dz).reshape(self.data.shape[0],
                                   self.data.shape[1],
                                   self.data.shape[2])
                       
        return self.__getvolumeobject_par(field,obj,dvol)[0]


    def volume(self,thres):
        """
        Computes the volume of stuff below the threshold
        
        Remember that the grid must be one point longer in each dimension!
        """                   
        
        return self.volume_object(self.data>thres)


    def all_volumes(self,objlist,objfield):
        """
        Returns a list of the volume of every single object in the field.
        """

        dx = self.x_grid[1:] - self.x_grid[:-1]
        dy = self.y_grid[1:] - self.y_grid[:-1]
        dz = self.z_grid[1:] - self.z_grid[:-1]

        dvol = np.kron(np.kron(dx,dy).reshape(self.data.shape[0],
                                              self.data.shape[1]),
                       dz).reshape(self.data.shape[0],
                                   self.data.shape[1],
                                   self.data.shape[2])

        # getvolumeobject =\
        #     lambda obj: ((objfield[1:-1,1:-1,1:-1] == obj).astype(np.int8)*
        #                  dvol).sum()
        getvolumeobject = lambda obj: self.__getvolumeobject_par(objfield,obj,dvol)

        return map(getvolumeobject,objlist)

        

    def get_object(self,objfield,obj):
        return (objfield == obj).astype(np.int8)


class SVORadix2(object):
    # def __power2check(self,n):
    #     return np.array([i%2 for i in primefactors(n)]).sum() == 0
    
    def __init__(self,NX):
        self.NX = NX
        self.tree = list()

    def __repr__(self):
        return '< Radix 2 SVO >'

    def pos(self,n):
        """
        n is a tuple with the tree coordinates
    
        8**len(n) must be equal to NX,this is kind of redundant, but you
        can play dirty tricks with this.
    
        Generate the position in the grid given the tree leaf coordinates.
    
        Remember that every leaf contains the coordinates to reach the
        level so the trick is decoding the coordinate for every leaf, that
        is an integer that goes from 0 to 7.
        """
        position = np.array([0,0,0,0,0,0], dtype=np.int)
        for (nit,level) in zip(n,count(1)):
            dist = self.NX/2**level
            facz = nit%2 
            facy = nit/2%2
            facx = nit/4%2

            position[0] = position[0]+facx*dist
            position[1] = position[0]+dist
            position[2] = position[2]+facy*dist
            position[3] = position[2]+dist
            position[4] = position[4]+facz*dist
            position[5] = position[4]+dist
    
        return position

    def grid_box(self,coords,x_grid,y_grid,z_grid):
        """Finds the position of the eight box corners"""
        position = self.pos(coords)
        return np.array(
            [[x_grid[position[0]],y_grid[position[2]],z_grid[position[4]]],
             [x_grid[position[0]],y_grid[position[2]],z_grid[position[5]]],
             [x_grid[position[0]],y_grid[position[3]],z_grid[position[4]]],
             [x_grid[position[0]],y_grid[position[3]],z_grid[position[5]]],
             [x_grid[position[1]],y_grid[position[2]],z_grid[position[4]]],
             [x_grid[position[1]],y_grid[position[2]],z_grid[position[5]]],
             [x_grid[position[1]],y_grid[position[3]],z_grid[position[4]]],
             [x_grid[position[1]],y_grid[position[3]],z_grid[position[5]]]]
            )
    
    def minmaxdist(self,coords,x_grid,y_grid,z_grid,x,y,z):
        """Gives the maximum and minimum euclidian distance of a box
        to a point"""
        distances = np.empty((8,),dtype=np.float)
        edges = self.grid_box(coords,x_grid,y_grid,z_grid)
        for (edge,i) in zip(edges,count()):
            distances[i] = (edge[0]-x)**2+\
                (edge[1]-y)**2 + (edge[2]-z)**2

        return (distances.min(),distances.max())

    def mindist_child(self,tree_level,x_grid,y_grid,z_grid,x,y,z,radius=1e10):
        """
        Given a list that is a clean tree level, obtain the
        candidates for the following level.

        Necessary to compute distances.
        
        Radius is a bounding radius in case the box count grows too much.
        """
        candidates = list()
        children = list()

        for (p,child) in [t for t in tree_level]:
            candidates.append(
                (self.minmaxdist(p,x_grid,y_grid,z_grid,x,y,z),child)
                )

        # minimum maximum distance
        minmaxdist = np.array([p[0][1] for p in candidates]).min()

        # if minimum distance is bigger than the minimum maximum, and
        # does not fit into the bounding shere, drop it
        for p in candidates:
            if p[0][0] <= minmaxdist and p[0][0] <= radius:
                children.append(p[1])

        # flatten the list
        return list(chain(*children))

    def mindist_voxels(self,x_grid,y_grid,z_grid,x,y,z):
        """
        Voxels that fulfill the minimum distance criteria with a given
        point
        """
        levels = int(np.log(self.NX-1)/np.log(2))+1
        
        # Generate the first clean level
        clean_level = map(self.tree[levels-3].__getitem__,
                          self.mindist_child(self.tree[levels-2],
                                            x_grid,
                                            y_grid,
                                            z_grid,
                                            x,y,z))

        for lev in range(levels-3,0,-1):
            clean_level = map(self.tree[lev-1].__getitem__,
                              self.mindist_child(clean_level,
                                                x_grid,y_grid,z_grid,
                                                x,y,z))

        return clean_level


    def eucdistance(self,f,coords,x_grid,y_grid,z_grid,xt,yt,zt):
        """
        Euclidean distance between a point and the closest point of the surface
        """

        NS = 10 #Number of voxel subdivisions

        # coords is coordinates in the tree
        position = self.pos(coords)

        # position is coordinates in the plane.
        f000 = f[position[0],position[2],position[4]]
        f001 = f[position[0],position[2],position[5]]
        f010 = f[position[0],position[3],position[4]]
        f011 = f[position[0],position[2],position[5]]
        f100 = f[position[1],position[2],position[4]]
        f101 = f[position[1],position[2],position[5]]
        f110 = f[position[1],position[3],position[4]]
        f111 = f[position[1],position[2],position[5]]
        fac0 = f000-f100
        fac1 = f000-f010-f100+f110
        fac2 = f000-f001-f100+f101
        fac3 = f000-f001-f010+f011-f100+f101+f110-f111
        fac4 = f000+f001+f010-f011

        # Solve equation for z in the trilinear interpolation
        z = lambda x,y: (fac0*x-(fac1*x-f000+f010)*y-f000)/(
            fac2*x-(fac3*x-fac4)*y-f000+f001)

        mindist = 1000
        minxpos = (0,0,0)

        #A lot of points
        for x,y in product(np.linspace(0.0,1.0,NS),repeat=2):
            zz = z(x,y)
            # Pick only the ones that fall within the voxel.
            if zz < 1.0 and zz > 0.0:
                scalx = x_grid[position[1]]-x_grid[position[0]]
                scaly = y_grid[position[3]]-y_grid[position[2]]
                scalz = z_grid[position[5]]-z_grid[position[4]]
                x0 = x_grid[position[0]]
                y0 = y_grid[position[2]]
                z0 = z_grid[position[4]]
                dist = (x*scalx+x0-xt)**2+(y*scaly+y0-yt)**2+(zz*scalz+z0-zt)**2
                if dist < mindist:
                    minxpos = (x*scalx+x0,y*scaly+y0,zz*scalz+z0)
                    mindist = dist

        return (minxpos,mindist)

    def eucdistance_c(self,f,coords,x_grid,y_grid,z_grid,xt,yt,zt):
        return fast_eucdistance(f,coords,self.NX,x_grid,y_grid,z_grid,xt,yt,zt)

    def eucdistance_numexpr(self,f,coords,x_grid,y_grid,z_grid,xt,yt,zt):
        """
        Euclidean distance between a point and the closest point of the surface
        """
        NS = 10 #Number of voxel subdivisions

        # coords is coordinates in the tree
        position = self.pos(coords)
        mindist = 1000
        minxpos = (0,0,0)

        # position is coordinates in the plane.
        f000 = f[position[0],position[2],position[4]]
        f001 = f[position[0],position[2],position[5]]
        f010 = f[position[0],position[3],position[4]]
        f011 = f[position[0],position[2],position[5]]
        f100 = f[position[1],position[2],position[4]]
        f101 = f[position[1],position[2],position[5]]
        f110 = f[position[1],position[3],position[4]]
        f111 = f[position[1],position[2],position[5]]
        fac0 = f000-f100
        fac1 = f000-f010-f100+f110
        fac2 = f000-f001-f100+f101
        fac3 = f000-f001-f010+f011-f100+f101+f110-f111
        fac4 = f000+f001+f010-f011

        x,y = np.meshgrid(np.linspace(0.0,1.0,NS),
                          np.linspace(0.0,1.0,NS))
        #numexpr
        z = ne.evaluate(
            "(fac0*x-(fac1*x-f000+f010)*y-f000)/(fac2*x-(fac3*x-fac4)*y-f000+f001)")
        
        scalx = x_grid[position[1]]-x_grid[position[0]]
        scaly = y_grid[position[3]]-y_grid[position[2]]
        scalz = z_grid[position[5]]-z_grid[position[4]]
        x0 = x_grid[position[0]]
        y0 = y_grid[position[2]]
        z0 = z_grid[position[4]]
        dist = ne.evaluate(
            '(x*scalx+x0-xt)**2+(y*scaly+y0-yt)**2+(z*scalz+z0-zt)**2')

        mdist= ma.masked_outside(dist,0.0,1.0)
        #Find minimum distance
        mindist = mdist.min()
        #ACHTUNG. These indices may be switched
        (miniy,minix) = np.unravel_index(mdist.argmin(),(NS,NS))
        minxpos = (x[miniy,minix]*scalx+x0,
                   y[miniy,minix]*scaly+y0,
                   z[miniy,minix]*scalz+z0)

        return (minxpos,mindist)

    def mindistance(self,f,otree,of,x_grid,y_grid,z_grid,n=0):
        """
        Minimum distance from the surface that the tree was built from
        and another surface

        The mesh must be the same.
        """
        mindistances = list()
        centroids = [leaf[0] for leaf in self.tree[0]]

        if n > 0:
            centroids = centroids[:n]

        for (coord,i) in zip([leaf[0] for leaf in self.tree[0]],count()):
            # Pick the centroid of the voxel
            (xt,yt,zt) = self.centroid(f,coord,x_grid,y_grid,z_grid)
            # Get the closest voxels from the other surface
            closest = otree.mindist_voxels(x_grid,y_grid,z_grid,xt,yt,zt)

            # Create the emtpy distances vector
            distances = list()
            for (ocoord,child) in closest:
                euctuple = otree.eucdistance_numexpr(
                    of,ocoord,x_grid,y_grid,z_grid,xt,yt,zt)
                # euctuple = otree.eucdistance(
                #     of,ocoord,x_grid,y_grid,z_grid,xt,yt,zt)
                # euctuple = otree.eucdistance_c(
                #     of,ocoord,x_grid,y_grid,z_grid,xt,yt,zt)
                distances.append(euctuple[1])

            mindist = np.min(ma.masked_invalid(distances))

            if i%1000 == 0:
                print "{}/{}. Distance over {} points".format(
                    i,len(self.tree[0]),len(closest)),
                print mindist

            mindistances.append(mindist)
    
        return np.array(mindistances)
               
    def centroid(self,f,coords,x_grid,y_grid,z_grid):
        """
        The centroid of a voxel is the point of the surface, that is
        linearly approximated inside the voxel, that is closer to the
        center of the voxel.  This point should be in the surface and
        as centered in the voxel as possible.

        It is therefore a good 0D estimation of where the surface is
        inside the voxel.
        """
        # coords is coordinates in the tree
        position = self.pos(coords)

        xt = 0.5*(x_grid[position[0]]+x_grid[position[1]])
        yt = 0.5*(y_grid[position[2]]+y_grid[position[3]])
        zt = 0.5*(z_grid[position[4]]+z_grid[position[5]])

        return self.eucdistance_numexpr(f,coords,x_grid,y_grid,z_grid,xt,yt,zt)[0]

    def center(self,f,coords,x_grid,y_grid,z_grid):
        """
        Returns the center of the voxel in physical grid
        """
        position = self.pos(coords)
        return (0.5*(x_grid[position[0]]+x_grid[position[1]]),
                0.5*(y_grid[position[2]]+y_grid[position[3]]),
                0.5*(z_grid[position[4]]+z_grid[position[5]]))

    def build(self,f):
        """Build the complete SVO reversally"""
        tic = clock()

        #Auxiliar variable for nodes
        levels = int(np.log(self.NX-1)/np.log(2))+1
        nodes = np.empty((8,),dtype=np.float32)
        
        # This builds the tree reversally.
        for (j,lev) in zip(range(levels-1,0,-1),count()):
            print "Building level {}/{}".format(lev,levels)
            voxels = list()                           
            if j==levels-1:
                for i in product(range(8),repeat=j):
                    select_lower(f,i,self.NX,voxels)
                ####
                #     Python implementation of select_lower if cython is
                #     not present.
                #
                #     position = self.pos(i)

                #     print position
                #     nodes[0] = f[position[0],position[2],position[4]]
                #     nodes[1] = f[position[0],position[2],position[5]]
                #     nodes[2] = f[position[0],position[3],position[4]]
                #     nodes[3] = f[position[0],position[3],position[5]]
                #     nodes[4] = f[position[1],position[2],position[4]]
                #     nodes[5] = f[position[1],position[2],position[5]]
                #     nodes[6] = f[position[1],position[3],position[4]]
                #     nodes[7] = f[position[1],position[3],position[5]]
    
                #     if not((np.sign(nodes)>0).all() or \
                #                (np.sign(nodes)<0).all()):
                #         voxels.append((i,[0]))

                # build_lower(f,j,self.NX,voxels)
    
            else:
                prev = self.tree[lev-1][0][0][:-1]
                voxidlist = []
                for (vox,voxid) in zip(self.tree[lev-1][1:],count()):
                    if vox[0][:-1] == prev:
                        voxidlist.append(voxid)
                    else:
                        voxidlist.append(voxid)
                        voxels.append((prev,voxidlist))
                        voxidlist = []
    
                    prev = vox[0][:-1]
    
                voxidlist.append(voxid+1)
                voxels.append((prev,voxidlist))
         
            self.tree.append(voxels)

        print "Tree built in {}s".format(clock()-tic)


def getenstro(filenameu,filenamev,filenamew,stats,NX0,NX,NZ):
    """Get enstrophy field from velocity components"""

    fu = tables.openFile(filenameu,'r')
    fv = tables.openFile(filenamev,'r')
    fw = tables.openFile(filenamew,'r')

    NZ2 = NZ/2
    NZ1 = NZ-2
    NY = stats.NY

    x_grid = stats.x
    xu = x_grid[:NX+1]
    xv = 0.5*(x_grid[:NX]+x_grid[1:NX+1])

    y_grid = stats.y
    yu = 0.5*(y_grid[:-1]+y_grid[1:])
    yv = y_grid[1:-1]

    lz = stats.dz
    z_grid = np.linspace(0,stats.dz,NZ1)[:NZ]

    u = np.empty((NX+1,NY+1,NZ2),dtype=np.complex64)
    v = np.empty((NX,NY,NZ2),dtype=np.complex64)
    w = np.empty((NX,NY+1,NZ2),dtype=np.complex64)
    dudzhat = np.empty((NX+1,NY+1,NZ2),dtype=np.complex64)
    dvdzhat = np.empty((NX,NY,NZ2),dtype=np.complex64)

    uphys = np.empty((NX+1,NY+1,NZ1),dtype=np.float32)
    dudzp = np.empty((NX+1,NY+1,NZ1),dtype=np.float32)

    vphys = np.empty((NX,NY,NZ1),dtype=np.float32)
    wphys = np.empty((NX,NY+1,NZ1),dtype=np.float32)

    # Derivatives
    dudy = np.empty((NX,NY,NZ1),dtype=np.float32)
    dudz = np.empty((NX,NY,NZ1),dtype=np.float32)

    dvdx = np.empty((NX,NY,NZ1),dtype=np.float32)
    dvdz = np.empty((NX,NY,NZ1),dtype=np.float32)

    dwdx = np.empty((NX,NY,NZ1),dtype=np.float32)
    dwdy = np.empty((NX,NY,NZ1),dtype=np.float32)

    # Auxiliar
    dudyp = np.empty((NX+1,NY+1),dtype=np.float32)
    dwdxp = np.empty((NX,NY+1),dtype=np.float32)


    print "Reading....."
    u.real[:,:,:] = fu.root.value[NX0:NX0+NX+1,:,::2]
    u.imag[:,:,:] = fu.root.value[NX0:NX0+NX+1,:,1::2]

    v.real[:,:,:] = fv.root.value[NX0:NX0+NX,:,::2]
    v.imag[:,:,:] = fv.root.value[NX0:NX0+NX,:,1::2]

    w.real[:,:,:] = fw.root.value[NX0:NX0+NX,:,::2]
    w.imag[:,:,:] = fw.root.value[NX0:NX0+NX,:,1::2]
    print "Read"


    print "fft-ing"
    u[:,:,0] = 0.0 + 0.0j
    v[:,:,0] = 0.0 + 0.0j
    w[:,:,0] = 0.0 + 0.0j
    # u[:,:,0] = u[:,:,0] - stats.ua[NX0:NX0+NX+1,:].astype(np.complex64)
    # v[:,:,0] = v[:,:,0] - stats.va[NX0:NX0+NX,:].astype(np.complex64)
    # w[:,:,0] = w[:,:,0] - stats.wa[NX0:NX0+NX,:].astype(np.complex64)

    kz = np.arange(NZ2)*2*np.pi/lz

    for i in range(NX+1):
        uphys[i,:,:] = np.fft.irfft(u[i,:,:],axis=1)[:,:NZ]*NZ
        for j in range(NY+1):
            dudzhat[i,j,:] = u[i,j,:]*kz

        dudzp[i,:,:] = np.fft.irfft(dudzhat[i,:,:],axis=1)[:,:NZ]*NZ

    for i in range(NX):
        wphys[i,:,:] = np.fft.irfft(w[i,:,:],axis=1)[:,:NZ]*NZ
        vphys[i,:,:] = np.fft.irfft(v[i,:,:],axis=1)[:,:NZ]*NZ
        for j in range(NY):
            dvdzhat[i,j,:] = v[i,j,:]*kz
            
        dvdz[i,:,:] = np.fft.irfft(dvdzhat[i,:,:],axis=1)[:,:NZ]*NZ


    print "Nothing left in phase space"

    del u
    del v
    del w
    del dudzhat
    del dvdzhat

    #dwdy
    #dudy

    for k in range(NZ1):
        if k%100 == 0: print k
        for i in range(NX):
            ev = interpolate.splrep(yu,wphys[i,:,k],k=5)
            dwdy[i,:,k] = interpolate.splev(yv,ev,der=1)

            ev = interpolate.splrep(yu,uphys[i,:,k],k=5)
            dudyp[i,:] = interpolate.splev(yu,ev,der=1)

    #dwdx
    #dvdx
        for j in range(NY+1):
            ev = interpolate.splrep(xv,wphys[:,j,k],k=5)
            dwdxp[:,j] = interpolate.splev(xv,ev,der=1)

        for j in range(NY):
            ev = interpolate.splrep(xv,vphys[:,j,k],k=5)
            dvdx[:,j,k] = interpolate.splev(xv,ev,der=1)

    #Interpolate what is needed
    
        ev = interpolate.RectBivariateSpline(xv,yu,dwdxp[:,:],kx=3,ky=3)
        dwdx[:,:,k] = ev(xv,yv)

        ev = interpolate.RectBivariateSpline(xu,yu,dudyp[:,:],kx=3,ky=3)
        dudy[:,:,k] = ev(xv,yv)

        ev = interpolate.RectBivariateSpline(xu,yu,dudzp[:,:,k],kx=3,ky=3)
        dudz[:,:,k] = ev(xv,yv)

    
    fu.close()
    fv.close()
    fw.close()
    
    enstro = (dwdy-dvdz)**2
    enstro = enstro + (dudz-dwdx)**2
    enstro = enstro + (dvdx-dudy)**2
    enstro = np.sqrt(enstro)

    return enstro


def getuvphys(filenameu,filenamev,stats,NX0,NX,NZ):
    """Get u and v velocity from its components"""
    fu = tables.openFile(filenameu,'r')
    fv = tables.openFile(filenamev,'r')

    #Now, for the field

    NZ2 = NZ/2
    NZ1 = NZ-2
    NY = stats.NY
    
    xu = stats.x[:NX+1]
    xv = 0.5*(xu[:NX]+xu[1:NX+1])

    yv = stats.yr
    yu = 0.5*(stats.y[:-1]+stats.y[1:])

    uucell = np.empty((NX+1,NY+1,NZ2),dtype=np.complex64)
    uvcell = np.empty((NX+1,NY,NZ2),dtype=np.complex64)
    u = np.empty((NX,NY,NZ2),dtype=np.complex64)
    v = np.empty((NX,NY,NZ2),dtype=np.complex64)
    uphys = np.empty((NX,NY,NZ1),dtype=np.float32)
    vphys = np.empty((NX,NY,NZ1),dtype=np.float32)

    print "Reading....."
    uucell.real[:,:,:] = fu.root.value[NX0:NX0+NX+1,:,::2]
    uucell.imag[:,:,:] = fu.root.value[NX0:NX0+NX+1,:,1::2]

    v.real[:,:,:] = fv.root.value[NX0:NX0+NX,:,::2]
    v.imag[:,:,:] = fv.root.value[NX0:NX0+NX,:,1::2]
    print "Read"

    fu.close()
    fv.close()

    # Interpolate u 

    print "Interpolating u"
    for k in range(NZ/2):
        if k%100 == 0: print k
        for i in range(NX+1):
            ev = interpolate.splrep(yu,uucell.real[i,:,k],k=5)
            uvcell.real[i,:,k] = interpolate.splev(yv,ev)

            ev = interpolate.splrep(yu,uucell.imag[i,:,k],k=5)
            uvcell.imag[i,:,k] = interpolate.splev(yv,ev)


        for j in range(NY):
            ev = interpolate.splrep(xu,uvcell.real[:,j,k],k=5)
            u.real[:,j,k] = interpolate.splev(xv,ev,der=1)

            ev = interpolate.splrep(xu,uvcell.imag[:,j,k],k=5)
            u.imag[:,j,k] = interpolate.splev(xv,ev,der=1)


    # Now I have u and v centered in the cell.
    # I substract the zero mode.

    #u[:,:,0] = u[:,:,0] - stats.ua[NX0:NX0+NX,:].astype(np.complex64)
    #v[:,:,0] = v[:,:,0] - stats.va[NX0:NX0+NX,:].astype(np.complex64)

    # u[:,:,0] = 0.0 + 0.0j
    # v[:,:,0] = 0.0 + 0.0j

    # And finally FFT.

    print "FFTing"
    for i in range(NX):
        uphys[i,:,:] = np.fft.irfft(u[i,:,:],axis=1)*NZ
        vphys[i,:,:] = np.fft.irfft(v[i,:,:],axis=1)*NZ

    return (uphys,vphys)
