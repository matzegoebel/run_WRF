# -*- coding: latin1 -*-
"""

Get pressure and temperature profiles according to ISA (ICAO / International(?)
Standard atmosphere).

Reference:
ICAO Doc 7488 (this module is based on the 1993 version)

Author(s)
---------
Martin Steinheimer, Austro Control (original version)
    TODO: Ask for Martin's permission to use / share this module.
Lukas Strauss, Austro Control (modifications)

"""

#-------------------------------------------------------------------------------
# MODULES
#-------------------------------------------------------------------------------
import math
import numpy as np

#-------------------------------------------------------------------------------
# CONSTANTS
#-------------------------------------------------------------------------------
g0 = 9.80665   # m/s2 - standard acceleration due to gravity
p0 = 101325.   # Pa - sea level pressure
R = 287.05287  # J/kg/K - specific gas constant
T0 = 288.15    # K - sea level temperature
kappa = 1.4    # = cp/cv - adiabatic index
rho0 = 1.225   # kg/m3 - sea level atmospheric density
a0 = 340.294   # m/s - speed of sound at sea level

# Define ISA layers according to the above reference.
# *h*, *T* and *p* are height, temperature and pressure at the lower bound of
# the layer, respectively, *gamma* is the temperature gradient in the layer.
# *h* used here corresponds to H (= geopotential altitude) in Doc 7488.
isa_layers = [ { "h": -5000., "T": 320.65, "p": 177687.,"gamma": -0.0065 },
               { "h": 0., "T": 288.15, "p": p0, "gamma": -0.0065 },
               { "h": 11000., "T": 216.65, "p": 22632., "gamma": 0. },
               { "h": 20000., "T": 216.65, "p": 5474.87, "gamma": 0.001 },
               { "h": 32000., "T": 228.65, "p": 868.014, "gamma": 0.0028 },
               { "h": 47000., "T": 270.65, "p": 110.906, "gamma": 0. },
               { "h": 51000., "T": 270.65, "p": 66.9384, "gamma": -0.0028 },
               { "h": 71000., "T": 214.65, "p": 3.95639, "gamma": -0.002 },
               { "h": 80000., "T": 196.65, "p": 0.886272, "gamma": np.nan }
             ]

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------
def height(p):
    """Calculate height for given pressure according to ISA.

    Parameters
    ----------
    p : float
        Pressure (Pa).

    Returns
    -------
    float
        Height (m MSL?).

    """
    if p < isa_layers[-1]["p"]:
        print ("isa.height requested for pressure lighter than minimum "
               "pressure supported:", p, "<", isa_layers[-1]["p"])
        return None

    i = 0
    while isa_layers[i]["p"] > p:
        i += 1
    i -= 1

    if isa_layers[i]["gamma"] == 0:
        # isotherm layer
        return (-R * isa_layers[i]["T"] / g0 *
                math.log( p / isa_layers[i]["p"] ) + isa_layers[i]["h"])
    else:
        expo = -R * isa_layers[i]["gamma"] / g0
        return ((isa_layers[i]["T"] * ( (p / isa_layers[i]["p"])**expo - 1)) /
                isa_layers[i]["gamma"] + isa_layers[i]["h"])

#-------------------------------------------------------------------------------
def pressure(z):
    """Calculate pressure for a given height according to ISA.

    Parameters
    ----------
    z : float
        Height (m MSL?):

    Returns
    -------
    float
        Pressure (Pa).

    """
    if z > isa_layers[-1]["h"]:
        print ("isa.pressure requested for height larger than supported:", z,
               ">", isa_layers[-1]["h"])
        return None

    i = 0
    while isa_layers[i]["h"] < z:
        i += 1
    i -= 1

    if isa_layers[i]["gamma"] == 0:
        # isotherm layer
        return (isa_layers[i]["p"] * math.exp( -g0 *
                ( z - isa_layers[i]["h"] ) / R / isa_layers[i]["T"] ))
    else:
        expo = -g0 / R / isa_layers[i]["gamma"]
        return (isa_layers[i]["p"] * ( 1 + isa_layers[i]["gamma"] *
                ( z - isa_layers[i]["h"] ) / isa_layers[i]["T"] )**expo)

#-------------------------------------------------------------------------------
def temperature(z, zisp=False):
    """Calculate temperature in ISA for given height or pressure.

    Parameters
    ----------
    z : float
        Height (m MSL?) or pressure (Pa), if *zisp* is True.
    zisp : bool, optional
        If True, *z* is pressure (Pa).

    Returns
    -------
    float
        Temperature (K).

    """
    if zisp:
        h = height(z)
    else:
        h = z

    if h > isa_layers[-1]["h"]:
        print ("isa.temperature requested for height larger than supported:",
               h, ">", isa_layers[-1]["h"])
        return None

    i = 0
    while isa_layers[i]["h"] < h:
        i += 1
    i -= 1

    if isa_layers[i]["gamma"] == 0:
        # isotherm layer
        return isa_layers[i]["T"]
    else:
        return (isa_layers[i]["T"] + isa_layers[i]["gamma"] *
                ( h - isa_layers[i]["h"] ))

