#%%
import pandas as pd
from pyarrow import parquet

dfPHQ = pd.read_parquet('/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/PHQ/allUsers_PHQdata.parquet', engine='pyarrow')
df = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/weekly_cluster_data.csv')

dictPHQ = dict(zip(dfPHQ['weekNumber'], dfPHQ['total']))

dfUser = dfPHQ.groupby(['userID'])
dictPHQ = dict()
for u, Ugroup in dfUser:
    dictWeek = dict()
    dfWk = Ugroup.groupby(['weekNumber'])
    for wk, Wgroup in dfWk:
        phqMax = Wgroup['total'].max()
        dictWeek[int(wk)] = int(phqMax)

    dictPHQ[int(u)] = dictWeek

phq_list = []
for i in range(len(df)):
    u = df['userID'][i]
    print('user: ' + str(u))
    w = df['weekNumber'][i]
    print('wk: ' + str(w))
    try:
        score = dictPHQ[u][w]
    except KeyError:
        print('keyerror')
        score = float('NaN')
    phq_list.append(score)
    print('===')

df['PHQ'] = phq_list
df.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/weekly_cluster_data_PHQscores.csv', index=False)


# %%

