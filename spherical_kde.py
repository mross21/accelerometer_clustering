#%%
import pandas as pd
from pyarrow import parquet
import numpy as np
import re
import glob
import scipy.optimize
from spherical_kde.utils import (cartesian_from_polar,
                                 polar_from_cartesian, logsinh,
                                 rotation_matrix)
from matplotlib import pyplot as plt
                                 
#%%
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def addSpherCoords(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['theta'] = np.arccos(z / (np.sqrt(x**2 + y**2 + z**2)))
    xyz['phi'] = np.arctan2(y, x) # arctan2 requires (y,x)
    return(xyz)


def VonMisesFisher_distribution(phi, theta, phi0, theta0, sigma0):
    """ Von-Mises Fisher distribution function.


    Parameters
    ----------
    phi, theta : float or array_like
        Spherical-polar coordinates to evaluate function at.

    phi0, theta0 : float or array-like
        Spherical-polar coordinates of the center of the distribution.

    sigma0 : float
        Width of the distribution.

    Returns
    -------
    float or array_like
        log-probability of the vonmises fisher distribution.

    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution
    """
    x = cartesian_from_polar(phi, theta)
    x0 = cartesian_from_polar(phi0, theta0)
    norm = -np.log(4*np.pi*sigma0**2) - logsinh(1./sigma0**2)
    return norm + np.tensordot(x, x0, axes=[[0], [0]])/sigma0**2

def VonMisesFisher_sample(phi0, theta0, sigma0, size=None):
    """ Draw a sample from the Von-Mises Fisher distribution.

    Parameters
    ----------
    phi0, theta0 : float or array-like
        Spherical-polar coordinates of the center of the distribution.

    sigma0 : float
        Width of the distribution.

    size : int, tuple, array-like
        number of samples to draw.

    Returns
    -------
    phi, theta : float or array_like
        Spherical-polar coordinates of sample from distribution.
    """
    n0 = cartesian_from_polar(phi0, theta0)
    M = rotation_matrix([0, 0, 1], n0)

    x = np.random.uniform(size=size)
    phi = np.random.uniform(size=size) * 2*np.pi
    theta = np.arccos(1 + sigma0**2 *
                         np.log(1 + (np.exp(-2/sigma0**2)-1) * x))
    n = cartesian_from_polar(phi, theta)

    x = M.dot(n)
    phi, theta = polar_from_cartesian(x)

    return phi, theta

def VonMises_mean(phi, theta):
    """ Von-Mises sample mean.

    Parameters
    ----------
    phi, theta : array-like
        Spherical-polar coordinate samples to compute mean from.

    Returns
    -------
    float

        ..math::
            \sum_i^N x_i / || \sum_i^N x_i ||

    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
    """
    x = cartesian_from_polar(phi, theta)
    S = np.sum(x, axis=-1)
    phi, theta = polar_from_cartesian(S)
    return phi, theta

def VonMises_std(phi, theta):
    """ Von-Mises sample standard deviation.

    Parameters
    ----------
    phi, theta : array-like
        Spherical-polar coordinate samples to compute mean from.

    Returns
    -------
        solution for

        ..math:: 1/tanh(x) - 1/x = R,

        where

        ..math:: R = || \sum_i^N x_i || / N

    Notes
    -----
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
        but re-parameterised for sigma rather than kappa.
    """
    x = cartesian_from_polar(phi, theta)
    S = np.sum(x, axis=-1)
    R = S.dot(S)**0.5/x.shape[-1]

    def f(s):
        return 1/np.tanh(s)-1./s-R

    kappa = scipy.optimize.brentq(f, 1e-8, 1e8)
    sigma = kappa**-0.5
    return sigma

#%%
pathAccel = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/accelerometer/test/'
pathOut = '/'

all_files = sorted(glob.glob(pathAccel + "*.parquet"), key = numericalSort)
for file in all_files:
    dfAccel = pd.read_parquet(file, engine='pyarrow')
    
    # convert cartesian coordinates to spherical
    dfAccel['x'] = pd.to_numeric(dfAccel['x'])
    dfAccel['y'] = pd.to_numeric(dfAccel['y'])
    dfAccel['z'] = pd.to_numeric(dfAccel['z'])
    dfSpher = addSpherCoords(dfAccel[:100])


    # vMF = VonMisesFisher_distribution(dfSpher['phi'], dfSpher['theta'], 0, 0, len(dfSpher))

    # # plot histogram of vMF
    # # Creating histogram
    # fig, ax = plt.subplots(figsize =(10, 7))
    # ax.hist(vMF)
    # # Show plot
    # plt.show()


print('finish')

# %%
import spherical_kde

sKDE = spherical_kde.SphericalKDE(dfSpher['phi'], dfSpher['theta'], weights=None, bandwidth=None, density=100)

import cartopy
fig=plt.figure()
ax = fig.add_subplot(111, projection=cartopy.crs.Mollweide())
sKDE.plot(ax)

# when trying to use full dataframe:
# MemoryError: Unable to allocate 36.8 GiB for an array with shape (10000, 493877) and data type float64

# %%
# find KL divergence for diff users





