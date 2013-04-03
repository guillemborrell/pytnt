from __future__ import print_function, division
import numpy as np
import tables
from scipy import interpolate


class MiniStats(object):
    def __init__(self, fname, rough=True):
        self.ROUGH = rough
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
        self.zr = None
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
        self.stats = tables.openFile(self.fname, mode='r')

        self.NX = self.stats.root.nx.read()[0]
        self.NY = self.stats.root.ny.read()[0]
        self.NZ = self.stats.root.nz2.read()[0]*2

        self.Re = self.stats.root.Re.read()[0]
        self.x  = np.linspace(0, self.stats.root.ax.read()*np.pi, self.NX)
        self.z  = np.linspace(0, self.stats.root.az.read()*np.pi*2, self.NZ)
        self.zr = np.linspace(0, self.stats.root.az.read()*np.pi*2, 3*self.NZ//2)
        self.y  = self.stats.root.y.read()
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

        vortza_w = self.stats.root.vortza.read()[:, 0]
        wstr = -1/self.Re*vortza_w

        if self.ROUGH:
            deltatau = np.empty(wstr.shape)
            gshape = 1.2*(-np.tanh((self.yr-0.13602)/0.036)+1)/(2.*18.)

            for i in range(len(wstr)):
                deltatau[i] = np.trapz(self.yr, gshape*self.ua[i, :])

            wstr += np.abs(deltatau)

        self.utau = np.sqrt(wstr)

        self.us = np.sqrt(np.abs(self.us - self.ua**2))
        self.vs = np.sqrt(np.abs(self.vs - self.va**2))
        self.ws = np.sqrt(np.abs(self.ws - self.wa**2))
        self.ps = np.sqrt(np.abs(self.ps - self.pa**2))
        self.uv = np.sqrt(np.abs(self.uv - self.ua*self.va))

    def ua_spline(self):
        return interpolate.RectBivariateSpline(self.x, self.yr, self.ua)

    def close(self):
        self.stats.close()

    def Ue(self,i=False):
        return np.mean(self.ua[:, -15:-5], axis=1)

    def theta(self,i=False):
        Ue = self.Ue()
        res = np.empty((self.NX,))
        for i in range(self.NX):
            res[i] = np.trapz(
                self.ua[i, :]/Ue[i]*(1-self.ua[i, :]/Ue[i]),
                self.yr)

        return res

    def deltastar(self,i=False):
        Ue = self.Ue()
        res = np.empty((self.NX,))
        for i in range(self.NX):
            res[i] = np.trapz((1-self.ua[i, :]/Ue[i]), self.yr)

        return res

    def delta99(self,i=False):
        if i:
            uint = interpolate.interp1d(self.ua[i, :].flatten(), self.yr)
            return uint(0.99)
        else:
            res = np.empty((self.NX,))
            for i in range(self.NX):
                uint = interpolate.interp1d(self.ua[i, :].flatten(), self.yr)
                res[i] = uint(0.99)
            return res

    def Reth(self,i=False):
        theta = self.theta()
        Ue = self.Ue()
        return theta*self.Re*Ue

    def Retau(self,i=False):
        if i:
            d99 = self.delta99(i)
            return d99*self.utau[i]*self.Re
        else:
            d99 = self.delta99()
            return d99*self.utau*self.Re

    def ydelta(self, i=False):
        if i:
            return self.yr/self.delta99()
        else:
            return self.yr/self.delta99(i)

    def xdelta(self, i=False):
        return self.x/self.delta99()[i]

    def zdelta(self, i=False):
        return self.z/self.delta99()[i]

    def load_budgets(self, budgets_file):
        self.budgets = tables.openFile(budgets_file, mode='r')

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
        dispu = (dispu - (dudx0**2 - dudy0**2))*(2/self.Re)
        dispv = (dispv - (dvdx0**2 - dvdy0**2))*(2/self.Re)
        dispw = (dispw - (dwdx0**2 - dwdy0**2))*(2/self.Re)
        dispuv = (dispuv - (dudx0*dvdx0 - dudy0*dvdy0))*(2/self.Re)

        #Wall units
        scab = 1/(self.utau**4 * self.Re)
        for i in range(self.NX):
            dispu[i, :] = dispu[i, :]*scab[i]
            dispv[i, :] = dispv[i, :]*scab[i]
            dispw[i, :] = dispw[i, :]*scab[i]
            dispuv[i, :] = dispuv[i, :]*scab[i]

        #Return only ke dissipation atm.
        return 0.5*(dispu+dispv+dispw)

    def production(self):
        dudx0 = self.budgets.root.dudx0.read()
        dudy0 = self.budgets.root.dudy0.read()
        dvdx0 = self.budgets.root.dvdx0.read()
        dvdy0 = self.budgets.root.dvdy0.read()

        produ = -2*self.us*dudx0 - 2*self.uv*dudy0
        prodv = -2*self.uv*dvdx0 - 2*self.vs*dvdy0
        produv = -self.us*dvdx0 - self.vs*dudy0

        #Wall units
        scab = 1/(self.utau**4 * self.Re)
        for i in range(self.NX):
            produ[i, :] = produ[i, :]*scab[i]
            prodv[i, :] = prodv[i, :]*scab[i]
            produv[i, :] = produv[i, :]*scab[i]

        return 0.5*(produ + prodv)
