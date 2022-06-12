#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:57:07 2022

@author: aleow
"""

"""
To generate 'num' points on a sphere of radius 'r' centred on the origin
- Random placement involves randomly chosen points for 'z' and 'phi'
- Regular placement involves chosing points such that there one point per d_area

References:
Deserno, 2004, How to generate equidistributed points on the surface of a sphere
http://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
"""

import random
import math
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

def random_on_sphere_points(r,num):
    points = []
    for i in range(0,num):
        z =  random.uniform(-r,r) 
        phi = random.uniform(0,2*math.pi)
        x = math.sqrt(r**2 - z**2)*math.cos(phi)
        y = math.sqrt(r**2 - z**2)*math.sin(phi)
        points.append([x,y,z])
        return points

def regular_on_sphere_points(r,num):
    points = []
    #Break out if zero points
    if num==0:
        return points
    a = 4.0 * math.pi*(r**2.0 / num)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta

    for m in range(0,m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0,m_phi):
            phi = 2.0 * math.pi * n / m_phi
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            points.append([x,y,z])
    return points


radius = 1
num = 5000

#print ("Randomly distributed points")
#random_surf_points = random_on_sphere_points(radius,points)
#pprint(random_surf_points)
#print " "

print ("Evenly distributed points")
regular_surf_points = regular_on_sphere_points(radius,num)


xyz=np.array(regular_surf_points)
x=xyz[:,0]
y=xyz[:,1]
z=xyz[:,2]


plt.plot(x, y, ',')
plt.axes().set_aspect('equal')
plt.show()

# %%
