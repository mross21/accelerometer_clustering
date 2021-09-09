from os import SF_NODISKIO
from typing import ValuesView
import pandas as pd
import numpy as np

#import
data = pd.read_csv(‘/User/theresanguyen/acceldata/unmasck_rawAccelData_5users.csv’)

#inspect structure
data.head()

#compute median and trim down
mdf = data.groupby(by=['userID','sessionNumber']).median().reset_index()
meandf = mdf.groupby(by=['userID','dayNumber']).mean().reset_index()
meandf

#get subject x timpoint matrix (z), subtract 1 bc indices start at 0 while data starts at 1
user = meandf['userID'].values - 1 
day = meandf['dayNumber'].values - 1 
Z = meandf['z'].values
#create matrix
mat = np.zeros(user.max() + 1, day.max() + 1)

for u,d,z in zip(user,day,Z): 
    mat[u,d] = z


import seaborn as sns 
sns.heatmap(mat)


## this doesn't work lol 
#c = 0 

#for i,u in enumerate(user): 
  #  for j,d in enumerate(day): 
   #     mat[u,d] = z[c]
   #     c = c + 1


#####compute avg of accel vector (the xyz columns into matrixes)
#xyz = mdf(['x','y','z']).values
#norm = np.zeros(xyz.shape[0])
#for i in range(xyz.shape[0]): 
    #norm[i]=np.linalg.norm(xyz[i,:])
#adding new column    
#mdf['accelNorm'] = norm
#np.var(norm)
#####jk norms are all ~1 

##compute avg 
#xyz = mdf(['x', 'y', 'z']).values
#np.mean(xyz, axis=1)
#np.var(xyz[:,2])

#installing python packages
#pip install seaborn 

