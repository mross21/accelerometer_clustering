import pandas as pd
import numpy as np
import re
import glob
import spherical_kde
from matplotlib import pyplot as plt
import math
from config import PATH_ACCELEROMETER_PARQUET, PATH_FIGURES
from io_utils import out_path
pd.options.mode.chained_assignment = None
                                 
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def accel_filter(xyz):
    x = pd.to_numeric(xyz['x'])
    y = pd.to_numeric(xyz['y'])
    z = pd.to_numeric(xyz['z'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    dfOut = xyz.loc[(xyz['r'] >= 0.95) & (xyz['r'] <= 1.05)]
    return(dfOut)

def addSpherCoords(xyz):
    x = pd.to_numeric(xyz['z'])
    y = pd.to_numeric(xyz['x'])
    z = pd.to_numeric(xyz['y'])
    xyz['r'] = np.sqrt(x**2 + y**2 + z**2)
    xyz['phi'] = round(np.mod(np.arctan2(y, x), np.pi*2),2) 
    xyz['theta'] = round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)),2) 
    return(xyz)

def regular_on_sphere_points(r,num):
    points = []
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

def main():
    all_files = sorted(glob.glob(PATH_ACCELEROMETER_PARQUET + "*.parquet"), key=numericalSort)

    radius = 1
    num = 1000
    regular_surf_points = regular_on_sphere_points(radius, num)
    pts_xyz = np.array(regular_surf_points)
    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]
    z = pts_xyz[:, 2]
    equi_phi = np.ndarray.round(np.mod(np.arctan2(y, x), np.pi * 2), 2)
    equi_theta = np.ndarray.round(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)), 2)

    for file in all_files:
        dfAccel = pd.read_parquet(file, engine='pyarrow')
        user = int(dfAccel['userID'].unique())
        print(user)

        dfAccel = accel_filter(dfAccel)
        addSpherCoords(dfAccel)

        userWks = list(dfAccel['weekNumber'].unique())
        for week in userWks:
            group = dfAccel.loc[dfAccel['weekNumber'] == week]

            sKDE = spherical_kde.SphericalKDE(group['phi'], group['theta'], weights=None, bandwidth=0.1)

            sKDE_densities = np.exp(sKDE(equi_phi, equi_theta))
            KDEdensities = np.column_stack([[user] * len(equi_phi), [week] * len(equi_phi), x, y, z, equi_phi, equi_theta, sKDE_densities])
            grpKDE = pd.DataFrame(KDEdensities, columns=['userID', 'weekNumber', 'z', 'x', 'y', 'phi', 'theta', 'density'])

            fig = plt.figure(facecolor=(1, 1, 1), figsize=(11, 11))
            plt.rcParams.update({'font.size': 18})
            ax = plt.axes(projection='3d')
            p = ax.scatter(grpKDE['x'], grpKDE['y'], grpKDE['z'], c=grpKDE['density'], cmap='viridis_r')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 27
            ax.zaxis.labelpad = 27
            ax.tick_params(axis='x', which='major', pad=25, rotation=45)
            ax.tick_params(axis='y', which='major', pad=15, rotation=-10)
            ax.tick_params(axis='z', which='major', pad=15, rotation=-10)
            ax.xaxis.set_major_locator(plt.MaxNLocator(9))
            ax.yaxis.set_major_locator(plt.MaxNLocator(9))
            ax.zaxis.set_major_locator(plt.MaxNLocator(9))
            cbar = plt.colorbar(p, fraction=0.03)
            cbar.set_label('Density')
            ax.view_init(-20, 280)
            plt.savefig(out_path(PATH_FIGURES, '3D_plot_user_' + str(user) + '_week_' + str(week) + '.png'))
            plt.close('all')


if __name__ == '__main__':
    main()
