## LABEL SESSION PHONE ORIENTATION

import pandas as pd
import numpy as np

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
    y = np.nanmedian(session['y'])
    z = np.nanmedian(session['z'])

    label = str('no_label')

    if y > 0:
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