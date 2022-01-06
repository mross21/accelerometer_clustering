#%%
import pandas as pd

jHoriz = '/home/mindy/Desktop/Horizontal_Session.json'

Horiz = pd.read_json(jHoriz, orient='index')
dfHoriz = pd.DataFrame(Horiz[0][1])

dfHoriz.to_csv('/home/mindy/Desktop/horiz_test_session.csv', index=False)

print('finish')
# %%
import json
jVert = '/home/mindy/Desktop/Vertical_Session.json'

with open(jVert, 'r') as j:
     contents = json.loads(j.read())
dfVert = pd.DataFrame(contents['accelerations'])
dfVert[['x', 'y', 'z']]
dfVert.to_csv('/home/mindy/Desktop/vert_test_session.csv', index=False)

print('finish')
# %%
