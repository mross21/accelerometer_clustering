#%%
import pandas as pd

file = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/flat_on_back.json'

j = pd.read_json(file, orient='index')
df = pd.DataFrame(j[0][1])

df.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/flat_on_back.csv', index=False)

print('finish')
# %%
import json
file = '/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/right_side.json'

with open(file, 'r') as j:
     contents = json.loads(j.read())
df = pd.DataFrame(contents['accelerations'])
df[['x', 'y', 'z']]
df.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/test_data/right_side.csv', index=False)

print('finish')
# %%
