import math
import re

import numpy as np
import pandas as pd
from math import asin, cos, sin, sqrt


numbers = re.compile(r"(\d+)")


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def accel_filter(xyz):
    x = pd.to_numeric(xyz["x"])
    y = pd.to_numeric(xyz["y"])
    z = pd.to_numeric(xyz["z"])
    xyz["r"] = np.sqrt(x**2 + y**2 + z**2)
    return xyz.loc[(xyz["r"] >= 0.95) & (xyz["r"] <= 1.05)]


def add_spher_coords(xyz):
    x = pd.to_numeric(xyz["z"])
    y = pd.to_numeric(xyz["x"])
    z = pd.to_numeric(xyz["y"])
    xyz["r"] = np.sqrt(x**2 + y**2 + z**2)
    xyz["phi"] = round(np.mod(np.arctan2(y, x), np.pi * 2), 2)
    xyz["theta"] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)), 2)
    return xyz


def regular_on_sphere_points(r, num):
    points = []
    if num == 0:
        return points
    a = 4.0 * math.pi * (r**2.0 / num)
    d = math.sqrt(a)
    m_theta = int(round(math.pi / d))
    d_theta = math.pi / m_theta
    d_phi = a / d_theta
    for m in range(0, m_theta):
        theta = math.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * math.pi * math.sin(theta) / d_phi))
        for n in range(0, m_phi):
            phi = 2.0 * math.pi * n / m_phi
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            points.append([x, y, z])
    return points


def haversine_dist(pt1, pt2):
    lat1, lng1 = pt1
    lat2, lng2 = pt2
    lat1 = lat1 - (np.pi / 2)
    lat2 = lat2 - (np.pi / 2)
    lng1 = lng1 - np.pi
    lng2 = lng2 - np.pi
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = (sin(lat * 0.5) ** 2) + (cos(lat1) * cos(lat2) * (sin(lng * 0.5) ** 2))
    return 2 * asin(sqrt(d))

