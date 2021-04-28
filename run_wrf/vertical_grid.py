# -*- coding: utf-8 -*-
"""

Create vertical grid in WRF using smooth stretching functions.

Authors
-------
Stefano Serafin and Matthias Göbel
    - stretching functions
Lukas Strauss and Matthias Göbel
    - plotting, testing

"""

import numpy as np
from scipy import integrate
import os
import matplotlib.pyplot as plt
import xarray as xr
import math

# %% grid creation methods

def linear_dz(ztop, dz0, dzmax=None, nz=None):
    """Create height levels with Linearly increasing dz from dz0 at z=0 to dzmax at z=ztop.
       Either nz or dzmax can be provided.

    Parameters
    ----------
    ztop : float
        top of the domain (m).
    dz0 : float
        grid spacing (m) in the lowest model layer.
    dzmax : float, optional
        Maxmimum grid spacing (m). Can be used instead of nz. The default is None.
    nz : int, optional
        Number of vertical levels. The default is None.

    Raises
    ------
    ValueError
        If neither nz nor dzmax are provided.

    Returns
    -------
    z : numpy array
        heights (m) of vertical levels.

    """
    stop = False
    search_nz = False
    if nz is None:
        if dzmax is None:
            raise ValueError("For vertical grid method 0: if nz is not defined, "
                             "dzmax must be defined!")
        nz = int(ztop/dzmax)
        search_nz = True

    while not stop:
        roots = np.roots((nz - 2)*[dz0] + [dz0-ztop])
        c = roots[~np.iscomplex(roots)].real
        c = float(c[c > 0])
        # if nz is not given, check if dzmax threshold is reached
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

    return z


def tanh_method(ztop, dzmin, dzmax=None, nz=None, D1=0, alpha=1):
    """
    Vertical grid with three layers. Spacing dz=dzmin in the first up to D1,
    then hyperbolic stretching up to D1+D2 and then constant again up to ztop.
    D2 is calculated automatically. If nz is None, nz is calculated from dzmax,
    while setting D2=ztop.

    Parameters
    ----------
    ztop : float
        domain top (m).
    dzmin : float
        spacing in the first layer (m).
    dzmax : float or None
        spacing in the third layer (m). If None, only two layers are used.
    nz : int
        number of levels or None. If None: 3rd layer is omitted.
    D1 : float
        depth of first layer (m).

    alpha : float, optional
        stretching coefficient. The default is 1.

    Returns
    -------
    z : numpy array of floats
        vertical levels.
    dz : numpy array of floats
        spacing for all levels.

    """
    n1 = D1/dzmin
    if n1 != int(n1):
        raise ValueError("Depth of layer 1 is not a multiple of its grid spacing!")
    n1 = int(n1)
    if nz is None:
        dzm = (dzmin + dzmax)/2
        n2 = math.ceil((ztop-D1)/dzm)
        # recalculate dzm and dzmax
        dzm = (ztop - D1)/n2
        dzmax = 2*dzm - dzmin
        nz = n1 + n2 + 1
        n3 = 0
    elif dzmax is None:  # only two layer
        # if nz is None:
        n2 = nz - n1 - 1
        dzm = (ztop - D1)/n2
        n3 = 0
    else:
        # average spacing in intermediate layer
        dzm = (dzmin + dzmax)/2

        # determine n2 from constraints
        n2 = round((ztop - D1 + (n1 - nz + 1)*dzmax)/(dzm-dzmax))
        D2 = dzm*n2
        n3 = nz - 1 - n2 - n1
        D3 = dzmax*n3
        ztop = D1 + D2 + D3
        nz = n1 + n2 + n3 + 1

        for i, n in enumerate((n2, n3)):
            if n != abs(int(n)):
                raise ValueError("Vertical grid creation failed!")

    # get spacing in layer 2 by stretching
    ind = np.arange(1, n2+1)
    a = (1 + n2)/2
    dz2 = dzm + (dzmin - dzm)/np.tanh(2*alpha)*np.tanh(2*alpha*(ind-a)/(1-a))

    # build spacings and levels
    dz = np.concatenate((np.repeat(dzmin, n1), dz2, np.repeat(dzmax, n3)))
    z = np.insert(np.cumsum(dz), 0, 0).astype(float)
    np.testing.assert_allclose(ztop, z[-1])

    return z, dz

# %% thermodynamic functions


def T_std(ztop, dz=1, strat=True):
    """Temperature (K) of the US standard atmosphere with
       Tsfc=15°C and dT/dz=6.5 K/km from sea level up to ztop
       with a spacing of dz (m).
       If strat=False, the tropospheric lapse rate is also used
       above 11 km, otherwise temperature is kept constant there
       with a quadratic smoothing between troposphere and
       stratosphere.
    """
    T0 = 15 + 273.15
    zvals = np.arange(0, ztop + 1, 1)
    T = T0 - 0.0065*zvals
    T = xr.DataArray(T, coords={"z": zvals}, dims=["z"])
    if (ztop > 11000) and strat:
        T.loc[11000:] = T.loc[11000]
        T.loc[10000:12000] = np.nan
        T = T.interpolate_na("z", method="quadratic")
    return T


def height_to_pressure_std(z, p0=1013.25, return_da=False, strat=True):
    """Convert height to pressure for the US standard atmosphere.
       If strat=False, the tropospheric lapse rate is also used
       above 11 km.
    """
    g = 9.81
    Rd = 287.06
    if np.array(z == 0).all():
        return p0
    ztop = np.array(z).max()
    T = T_std(ztop, strat=strat)
    T_int = integrate.cumtrapz(g / (Rd * T), T.z)
    T_int = np.insert(T_int, 0, 0)
    p = T.copy()
    p[:] = p0*np.exp(-T_int)
    p = p.interp(z=z)
    if not return_da:
        p = p.values

    return p


# %% create levels


def create_levels(ztop, dz0, method=0, nz=None, dzmax=None, theta=None, p0=1000,
                  table=False, plot=False, savefig=False, strat=False, **kwargs):
    """Create eta values for the vertical grid in WRF.

    First the metric heights of the model levels are calculated from the input parameters.
    Then these are converted to pressure levels using the US standard atmosphere.
    Finally the corresponding eta values are computed.

    Parameters
    ----------
    ztop : float
        top of the domain (m).
    dz0 : float
        grid spacing (m) in the lowest model layer.
    method : int, optional
        Grid creation method:
        0: linearly increasing dz from dz0 at z=0 to dzmax at z=ztop (default)
        1: ARPS 3-layer tanh method
    nz : int, optional
        Number of vertical levels. The default is None.
    dzmax : float, optional
        Maxmimum grid spacing (m). Can be used instead of nz for methods 0 and 3
        and is optional for method 2. The default is None.
    p0 : float, optional
        surface pressure (hPa). The default is 1000.
    table : bool, optional
        Print table. The default is False.
    plot : bool, optional
        Plot the vertical grid. The default is False.
    savefig : bool, optional
        Save the plot. The default is False.
    strat : bool, optional
        Take stratosphere into account when calculating pressure from height.
        The default is False.
    **kwargs :
        Keyword arguments passed to the underlying grid creation method.


    Returns
    -------
    eta : numpy array
        eta values to be used in WRF.
    dz : numpy array
        vertical grid spacings (m).

    """
    if "figloc" in kwargs:
        figloc = kwargs.pop("figloc")
    else:
        figloc = "~/"

    if method == 0:
        # linearly increasing dz from dz0 at z=0 to dzmax at z=ztop
        z = linear_dz(ztop, dz0, dzmax=dzmax, nz=nz)
    elif method == 1:
        # ARPS tanh method
        z, _ = tanh_method(ztop, dz0, dzmax=dzmax, nz=nz, **kwargs)
    else:
        raise ValueError("Vertical grid method {} not implemented!".format(method))

    # convert height to pressure levels
    ptop = height_to_pressure_std(ztop, p0=p0, strat=strat)
    p = height_to_pressure_std(z, p0=p0, strat=strat)

    if (method != 1) and (np.round(z[-1], 3) != ztop):
        raise ValueError("Uppermost level ({}) is not at ztop ({})!".format(z[-1], ztop))

    # Define stretched grid in pressure-based eta coordinate.
    eta = (p - ptop) / (p0 - ptop)
    # scale to 0-1 range
    eta = (eta - eta.min())/(eta.max() - eta.min())

    # Compute dp, dz and the alphas
    dp = np.diff(p)
    dp = np.append(np.nan, dp)
    dz = z[1:] - z[:-1]
    dz = np.append(dz, np.nan)
    alpha = np.diff(eta)[1:] / np.diff(eta)[:-1]
    alpha = np.append(np.append(np.nan, alpha), np.nan)
    alpha_z = np.diff(z)[1:] / np.diff(z)[:-1]
    alpha_z = np.append(np.append(np.nan, alpha_z), np.nan)

    if any(alpha_z > 1.1):
        print("WARNING: vertical grid stretching ratio exceeds 110 % for some vertical levels!")
    # ---------------------------------------------------------------------------
    # Make a plot.
    # ---------------------------------------------------------------------------

    # Define some reference heights to be drawn in the figure.
    zPBLref = 1000
    zTPref = 11000
    if plot:
        fig, ax1a = plt.subplots(figsize=(5, 4))
        ms = 2
        # z
        ax1a.plot(dz, z, 'ko', ms=ms)
        ax1a.set_xlim(0, np.nanmax(dz)+20)
        ax1a.grid(c=(0.8, 0.8, 0.8))
        ax1a.set_ylabel('height (m)')
        ax1a.set_xlabel('$\Delta z$ (m)')

        ax1b = ax1a.twiny()
        ax1b.plot(alpha_z, z, 'o', c="blue", ms=ms)
        xlabel = "$\Delta z (i)/\Delta z (i-1)$ (blue), $\Delta \eta (i)/\Delta \eta (i-1)$ (red)"
        ax1b.set_xlabel(xlabel)
        ax1b.plot(alpha, z, 'o', c="red", ms=ms)

        for ax in [ax1a, ax1b]:
            ax.set_ylim(0, max(z))

        if savefig:
            figloc = os.path.expanduser(figloc)
            fig.savefig(figloc + '/wrf_stretched_grid_etaz.pdf')

    # ---------------------------------------------------------------------------
    # Print vertical grid data.
    # ---------------------------------------------------------------------------
    printedPBL = False
    printedTP = False
    if table:
        header = ('|  ml |    eta | p (hPa) | z (m) | -dp (hPa) | dz (m) | '
                  'alpha | alpha_z |')
        print('-'*len(header))
        print('|    With a surface pressure of %7.2f hPa' % (p0))
        print('|   and a model-top pressure of %7.2f hPa' % (ptop))
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for i in range(p.size):
            if z[i] > zPBLref and not printedPBL:
                print('|%s|' % ('-'*(len(header)-2)))
                printedPBL = True
            if z[i] > zTPref and not printedTP:
                print('|%s|' % ('-'*(len(header)-2)))
                printedTP = True
            print(('| %3i | % 5.3f | %7.0f | %5.0f | %9.1f | %6.0f | %5.3f '
                   '| %7.3f |' %
                   (i, eta[i], p[i], z[i], -dp[i], dz[i], alpha[i],
                    alpha_z[i])))
        print('-'*len(header))

    return eta, dz


# %%
if __name__ == '__main__':
    p0 = 1000
    ztop = 15000
    eta, dz = create_levels(ztop=ztop, dz0=20, method=0, dzmax=400, D1=0, alpha=1.,
                            p0=p0, table=True, plot=True, savefig=True, strat=False)

    print(', '.join(['%.6f' % eta_tmp for eta_tmp in eta]))
