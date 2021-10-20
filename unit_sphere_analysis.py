#%%
import pandas as pd
from pyarrow import parquet
import numpy as np

# https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion

def appendSphericalCoords(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
    
#%%
# test function
pts = np.random.rand(3000000, 3)
ptsSpher = appendSphericalCoords(pts)



# %%
# sampling from von mises-fisher distribution
# https://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python

# estimating kappa for von mises distribution
# https://stats.stackexchange.com/questions/18692/estimating-kappa-of-von-mises-distribution


#%%

# functions for the spherical vMF distribution 
# https://spherical-kde.readthedocs.io/en/latest/_modules/spherical_kde/distributions.html 
    # need to 

# 1. for vMF distribution, estimate kappa and mu
    # https://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
    # x is d-dimensional unit random vector
    # magnitude(mu) = 1
    # magnitude(r) = magnitude(sum(xi))
    # mu = r/magnitude(r) = sum(xi)/magnitude(sum(xi))
    # r' = magnitude(r)/n
    # d = number of dimensions of data
    # k = (r'd - r'^3)/(1-r'^2)

# 2. calculate vMF KDE 
    # https://people.sc.fsu.edu/~jburkardt/cpp_src/sphere_lebedev_rule/sphere_lebedev_rule.html 
    # do we need kappa and mu?
    # uses polar coordinates phi and theta

# 3. Lebedev quadrature for KL?
    # https://people.sc.fsu.edu/~jburkardt/cpp_src/sphere_lebedev_rule/sphere_lebedev_rule.html 
    # https://en.wikipedia.org/wiki/Lebedev_quadrature 
    # python version?:
        # https://github.com/Rufflewind/lebedev_laikov/blob/master/lebedev_laikov.py





