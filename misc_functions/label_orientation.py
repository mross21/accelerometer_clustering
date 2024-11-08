"""
@author: Mindy Ross
python version 3.7.4
pandas version: 1.3.5
numpy version: 1.19.2
"""
#%%
## LABEL SESSION PHONE ORIENTATION
# based on median XYZ values

import pandas as pd
import numpy as np

# label session orientation
def label_orientation(session):
    # Possible orientations:
        # upright
            # x: [-0.5, 0.5]
            # y: [-1, -0.5]
            # z: [-0.5, 0.5]
        # face down
            # x: [-0.5, 0.5]
            # y: [-0.5, 0]
            # z: [0.5, 1]
        # face up
            # x: [-0.5, 0.5]
            # y: [-0.5, 0]
            # z: [-1, -0.5]
        # horizontal left
            # x: [-1, -0.5]
            # y: [-0.5, 0]
            # z: [-0.5, 0.5]
        # horizontal right
            # x: [0.5, 1]
            # y: [-0.5, 0]
            # z: [-0.5, 0.5]
        # upside down
            # x: [-1, 1]
            # y: [0 , 1] 
            # z: [-1, 1]
    
    x = np.nanmedian(session['x'])
    print('x: ' + str(x))
    y = np.nanmedian(session['y'])
    print('y: ' + str(y))
    z = np.nanmedian(session['z'])
    print('z: ' + str(z))

    label = str('no_label')

    if y > 0.1:
        label = str('upside_down')
    if y <= -0.5:
        label = str('upright')
    if z >= 0.5:
        label = str('face_down')
    if z <= -0.5: 
        label = str('face_up')
    if x <= -0.5:
        label = str('horizontal_left')
    if x >= 0.5:
        label = str('horizontal_right')
    if label == 'no_label':
        print('session has no orientation label')
    return(label)

# label session as horizontal or not horizontal
def binary_orientation(session):
    x = np.nanmedian(session['x'])
    if abs(x) >= 0.5:
        label = str('horizontal')
    else:
        label = str('not_horizontal')
    return(label)
