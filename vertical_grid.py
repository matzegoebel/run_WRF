# -*- coding: utf-8 -*-
"""

Generate stretched vertical grid for WRF using a smooth variation in the
stretching in eta. Two different types of grids can be constructed, a three-
layer and a two-layer grid. See the code for more.

Authors
-------
Stefano Serafin
    - stretching functions
Lukas Strauss
    - plotting, testing

"""

#-------------------------------------------------------------------------------
# MODULES
#-------------------------------------------------------------------------------
# Import modules.
import numpy as np
import matplotlib.pyplot as plt
from metpy import calc as metcalc
from metpy.units import units as metunits
from metpy import constants as metconst
import scipy as sp
import tools
import xarray as xr

# Import user modules.
import os

figloc = "~"
figloc = os.path.expanduser(figloc)
#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def pressure_from_theta(theta, p0=1e5):
    """Calculate pressure from potential temperature and surface pressure"""
    cp = metconst.Cp_d
    g = metconst.g
    c = (metconst.Cp_d/metconst.Rd).m/1000
    integral = -sp.integrate.cumtrapz(1/theta.values, theta.level.values)*metunits.meter/metunits.K
    p = theta.copy()
    if "unit" in dir(p0):
        units = p0.units
    else:
        units = ""
    p = p.assign_attrs(units=units)
    pt = p0*(g/cp*integral + 1)**c
    p[:] =  np.concatenate(([p0],pt))

    return p

def strheta_1(nlev, eta1, eta2, deta0, n2):
    """Generate a three-layer grid.

    The grid spacing is constant in the first and third layer.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    eta1 : float
        Bottom of the second layer.
    eta2 : float
        Bottom of the third layer.
    deta0 : float
        Grid spacing in eta in the first layer.
    n2 : int
        Number of levels in the third layer.

    Returns
    -------
    eta : np.ndarray
        Eta levels.
    deta : np.ndarray
        Eta level spacings.

    """

    # Determine the number of grid points in the two lowest layers,
    # numbered 0-1-2 bottom to top

    n0 = int((1.-eta1)/deta0)
    n1 = int(nlev - n0 - n2 - 1)

    # The delta eta in the upper layer is necessarily determined by the
    # stretching function in the intermediate layer (see below)

    deta2 = 2.*(eta1-eta2)/n1-deta0

    # Pre-allocate arrays for eta and delta eta

    eta  = np.ones([nlev])
    deta = np.ones([nlev-1])*deta0

    # Layer 0 (bottom): constant deta = deta0

    for i in range(n0):
      deta[i] = deta0
      eta[i+1] = eta[i]-deta[i]

    # Layer 1 (intermediate): deta stretches with a sinusoidal function.
    # Consequently, the stretching factor d(deta)/dj (j=grid point index)
    # is a cosine square function, which varies smoothly and has a maximum in
    # the middle of the stretching layer.
    # The amplitude of the cosine^2 function that describes the stretching
    # is dictated by eta1, eta2, deta0 and the number of levels in the
    # stretching layer.
    # It can be demonstrated that deta stretches from deta0 to deta2, with
    # deta2 = 2.*(eta1-eta2)/n1-deta0
    # Consequence: there is no guarantee that deta2 = eta2/n2.

    ampl = 4*(eta1-eta2-n1*deta0)/(n1**2)
    for i in range(n0,n0+n1):
      j = i-n0
      deta[i] = deta0 + ampl*(j/2.-n1/(4.*np.pi)*np.sin(2.*np.pi*j/n1))
      eta[i+1] = eta[i]-deta[i]

    # Check that everything is ok, that is, the deta in the uppermost
    # layer (detadum) is exactly as expected (deta2)

    #j = n1
    #detadum = deta0 + ampl*(j/2.-n1/(4.*np.pi)*np.sin(2.*np.pi*j/n1))
    #print deta2, detadum

    # Layer 2 (top): constant deta = deta2

    for i in range(n0+n1,nlev-1):
      deta[i] = deta2
      eta[i+1] = eta[i]-deta[i]

    # Generally, the coordinate of the uppermost level eta.min() will not be
    # exactly zero. Therefore, rescale eta so that its range is exactly 0-1
    # and recompute deta accordingly

    eta = (eta-eta.min())/(eta.max()-eta.min())
    deta = -eta[1:]+eta[:-1]

    return eta, deta

def strheta_2(nlev, eta1, deta0):
    """Generate a two-layer grid.

    The grid spacing is constant in the first layer and increases in the second.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    eta1 : float
        Bottom of the second layer.
    deta0 : float
        Grid spacing in eta in the first layer.

    Returns
    -------
    eta : np.ndarray
        Eta levels.
    deta : np.ndarray
        Eta level spacings.

    """

    # Determine the number of grid points in the two layers,
    # numbered 0-1 bottom to top

    n0 = int((1.-eta1)/deta0)
    n1 = int(nlev - n0 - 1)

    # The delta eta at the model top, that is, for the first
    # hypothetical layer above model top, is necessarily determined
    # by the stretching function (see below)

    detamax = deta0+2.*np.pi**2./(np.pi**2.-4.)*(eta1-deta0*n1)/n1

    # Pre-allocate arrays for eta and delta eta

    eta  = np.ones([nlev])
    deta = np.ones([nlev-1])*deta0

    # Layer 0 (bottom): constant deta = deta0

    for i in range(n0):
      deta[i] = deta0
      eta[i+1] = eta[i]-deta[i]

    # Layer 1: deta stretches with a sinusoidal function.
    # Consequently, the stretching factor d(deta)/dj (j=grid point index)
    # is a cosine square function, which varies smoothly and has a maximum
    # at the domain top.
    # The amplitude of the cosine^2 function that describes the stretching
    # is dictated by eta1, deta0 and the number of levels in the
    # stretching layer.

    ampl = 4.*np.pi**2./(np.pi**2.-4.)*(eta1-n1*deta0)/(n1**2.)
    for i in range(n0,nlev-1):
      j = i-n0
      deta[i] = deta0 + ampl*(j/2.-n1/(2.*np.pi)*np.sin(np.pi*j/n1))
      eta[i+1] = eta[i]-deta[i]

    # Check that everything is ok, that is, the deta at the model
    # top (detadum) is exactly as expected (detamax)

    #j=nlev-1-n0
    #detadum = deta0 + ampl*(j/2.-n1/(2.*np.pi)*np.sin(np.pi*j/n1))
    #print detamax, detadum

    # Generally, the coordinate of the uppermost level eta.min() will not be
    # exactly zero. Therefore, rescale eta so that its range is exactly 0-1
    # and recompute deta accordingly

    eta = (eta-eta.min())/(eta.max()-eta.min())
    deta = -eta[1:]+eta[:-1]

    return eta,deta

def tanh_method(nz, dz0, dzmax=200, alpha=0.5):
    ind = np.arange(1, nz+1)
    a = (1+nz)/2
    dzm = (dzmax+dz0)/2
    dz = dzm + (dz0 - dzm)/np.tanh(2*alpha)*np.tanh(2*alpha*(ind-a)/(1-a))
    z = np.cumsum(dz)

    return z


def create_levels(ztop, dz0, method=0, nz=None, dzmax=None, etaz1=None, etaz2=None, n2=None, theta=None, p0=None, plot=True, table=True, savefig=False, imgtype="pdf"):
#for method 0 (linearly increasing dz from dz0 at z=z0 to dzt at z=ztop)
   # dzt = 200
#for method 1 (ARPS method)
#        detaz0 = 0.0008
#        etaz1 = 0.999
# for method 2:
        #schmidli:
#        etaz1 = 0.87
#        etaz2 = 0.4
#        detaz0 = 0.0038
#        nz = 143
#        n2 = 37
    z0 = 0
    if method == 0: # linearly increasing dz from dz0 at z=z0 to dzt at z=ztop
        stop = False
        search_nz = False
        if nz is None:
            if dzmax is None:
                raise ValueError("For vertical grid method 0: if nz is not defined, dzmax must be defined!")
            nz = int(ztop/dzmax)
            search_nz = True

        while not stop:
            roots = np.roots((nz - 2)*[dz0]+ [dz0-ztop])
            c = roots[~np.iscomplex(roots)].real
            c = float(c[c > 0])
            #if nz is not given, check if dzmax threshold is reached
            if search_nz:
                dzmax_c = dz0*c**(nz-2)
                if dzmax_c <= dzmax:
                    stop = True
            else:
                stop = True
            if not stop:
                nz += 1

        z = np.zeros(nz)
        for i in range(nz - 1):
            z[i+1] = dz0 + z[i] * c

    elif method == 1: # 2- layer
        if any([arg is None for arg in [nz,etaz1]]):
            raise ValueError("For vertical grid method 1, nz and etaz1 must be defined!")
        detaz0 = dz0/(ztop - dz0)
        etaz, detaz = strheta_2(nz, etaz1, detaz0)
        z = ztop + etaz * (z0 - ztop)
    elif method == 2:  # ARPS method 3-layer
        if any([arg is None for arg in [nz,etaz1,etaz2,n2]]):
            raise ValueError("For vertical grid method 2, nz, etaz1, etaz2, and n2 must be defined!")
        detaz0 = dz0/(ztop - dz0)
        etaz, detaz = strheta_1(nz, etaz1, etaz2, detaz0, n2)
        z = ztop + etaz * (z0 - ztop)

    elif method == 3:
        z = tanh_method(nz, dz0, dzmax)

    if theta is None:
       # ptop = isa.pressure(ztop)
       # p = np.array(list(map(isa.pressure, z)))
        ptop = metcalc.height_to_pressure_std(ztop*metunits.m).m*100
        p = metcalc.height_to_pressure_std(z*metunits.m).m*100
    else:
        pth = pressure_from_theta(theta, p0=p0)
        p = pth.interp(level=z, kwargs=dict(fill_value="extrapolate")).values
        ptop = pth.interp(level=ztop, kwargs=dict(fill_value="extrapolate")).values

    psfc = p.max()
    # Define stretched grid in pressure-based eta coordinate.
    eta = (p - ptop) / (psfc-ptop)
    eta[0] = 1
    eta[-1] = 0

    # Compute dp, dz and the alphas

    dp = np.diff(p)
    dp = np.append(np.nan, dp)
    dz = z[1:] - z[:-1]
    dz = np.append(dz,np.nan)
    alpha = np.diff(eta)[1:] / np.diff(eta)[:-1]
    alpha = np.append(np.append(np.nan,alpha),np.nan)
    alpha_z = np.diff(z)[1:] / np.diff(z)[:-1]
    alpha_z = np.append(np.append(np.nan,alpha_z),np.nan)
    eps = 1e-6

    #---------------------------------------------------------------------------
    # Make a plot.
    #---------------------------------------------------------------------------

       # Define some reference heights to be drawn in the figure.
    #zPBLref = z0 + 1500
    zPBLref = z0 + 1000
    zTPref = 11000
    if plot:
        fig, axes = plt.subplots(ncols=2, figsize=(16, 13), sharey=True)
        # z
        ax1a = axes[0]
        ax1a.plot(dz, z, 'bo', ms=4)
        ax1a.set_xlim(0, 500)
        ax1a.grid(c=(0.8, 0.8, 0.8))
        ax1a.set_ylabel('z (m MSL)')
        ax1a.set_xlabel('dz (m)', color="b")

        ax1b = ax1a.twiny()
        ax1b.plot(alpha_z, z, 'go', ms=4)
        ax1b.set_xlim(0.95, 1.15)
        ax1b.set_xlabel('dz(n) / dz(n-1)', color="g")
        # alpha_eta
        ax2a = axes[1]
        ax2a.plot(alpha, z, 'bo', ms=4)
        ax2a.set_xlim(0.8, 1.35)
        ax2a.grid(c=(0.8, 0.8, 0.8))
        ax2a.set_ylabel('z (m MSL)')
        ax2a.set_xlabel('eta stretching factor alpha')

        for ax in [ax1a, ax1b, ax2a]:
            ax.set_ylim(0, max(z))
#            ax.axhline(z1, ls='--', c=(0.9, 0.9, 0.9))
            ax.axhline(zPBLref, ls=':', c='k')
            ax.axhline(zTPref, ls=':', c='k')
#            ax.axhline(z2, ls='--', c=(0.9, 0.9, 0.9))

        # Save figure.
        if savefig:
            fig.savefig(figloc + 'wrf_stretched_grid_etaz.%s'%imgtype)

    #---------------------------------------------------------------------------
    # Print vertical grid data.
    #---------------------------------------------------------------------------
    printedPBL = False
    printedTP = False
    if table:
        header =  ('|  ml |    eta | p (hPa) | z (m) | -dp (hPa) | dz (m) | '
                   'alpha | alpha_z |')
        print('-'*len(header))
        print('|    With a surface pressure of %7.2f hPa'%(1e-2*psfc))
        print('|   and a model-top pressure of %7.2f hPa'%(1e-2*ptop))
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for i in range(p.size):
            if z[i] > zPBLref and not printedPBL:
                print('|%s|'%('-'*(len(header)-2)))
                printedPBL = True
            if z[i] > zTPref and not printedTP:
                print('|%s|'%('-'*(len(header)-2)))
                printedTP = True
            print(('| %3i | % 5.3f | %7.0f | %5.0f | %9.1f | %6.0f | %5.3f '
                   '| %7.3f |'%
                   (i, eta[i], 1e-2*p[i], z[i], -1e-2*dp[i], dz[i], alpha[i],
                    alpha_z[i])))
        print('-'*len(header))



    return eta, dz


if __name__ == '__main__':
    p0 = 1e5
    slope_t, intercept_t = 0.004, 296#0.004, 293
    levels = np.arange(0, 12001, 20)
    theta = xr.DataArray(dims=["level"], coords={"level" : levels})
    theta["level"] = theta.level.assign_attrs(units="m")
    theta = theta.level * slope_t + intercept_t
    theta = theta.assign_attrs(units="K")
    # p = pressure_from_theta(theta, p0=p0)
    # p.interp(level=z)
   # eta, dz = create_levels(nz=160, ztop=12000, method=0, dz0=20, etaz1=0.87, etaz2=0.4, n2=37, theta=theta,p0=p0, plot=True, table=True, savefig=False)
   # eta, dz = create_levels(ztop=5000, method=0, dz0=25, dzmax=200, theta=theta,p0=p0)
    eta, dz = create_levels(ztop=12200, dz0=20, method=3, nz=20, dzmax=200, theta=theta, p0=p0)

    print(', '.join(['%.6f'%eta_tmp for eta_tmp in eta]))

