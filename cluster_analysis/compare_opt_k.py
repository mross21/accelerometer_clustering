#%% 
import pandas as pd

df1 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_noRot_cubicinterp.csv', index_col=False, header=None)
df2 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_rot_cubicinterp.csv', index_col=False, header=None)
df3 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_rot_lininterp.csv', index_col=False, header=None)
# df4 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_XZ_noRot_cubicinterp.csv', index_col=False, header=None)
df5 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_XZ_rot_cubicinterp.csv', index_col=False, header=None)
df6 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_rot_lininterp_fminsearch.csv', index_col=False, header=None)

df1.columns = ['userID','weekNumber','k1']
df2.columns = ['userID','weekNumber','k2']
df3.columns = ['userID','weekNumber','k3']
# df4.columns = ['userID','weekNumber','k4']
df5.columns = ['userID','weekNumber','k5']
df6.columns = ['userID','weekNumber','k6']


# df = pd.concat([df1,df2,df3,df5]).drop_duplicates(keep=False)


df12 = pd.merge(df1,df2, on = ['userID','weekNumber'],how='outer')
df123 = pd.merge(df12,df3, on = ['userID','weekNumber'],how='outer')
# df1234 = pd.merge(df123,df4, on = ['userID','weekNumber'],how='outer')
df12345 = pd.merge(df123,df5, on = ['userID','weekNumber'],how='outer')
df = pd.merge(df12345,df6, on = ['userID','weekNumber'],how='outer')

df['k_mode'] = df[['k1','k2','k3','k5','k6']].mode(axis=1).iloc[:, 0]

df.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_freq.csv',index=False)

# %%

dfr1 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_noRot_cubicinterp.csv', index_col=False, header=None)
dfr2 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_tRot_cubicinterp.csv', index_col=False, header=None)
dfr3 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_pRot_cubicinterp.csv', index_col=False, header=None)
dfr4 = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_pt_ptRot_cubicinterp.csv', index_col=False, header=None)

dfr1.columns = ['userID','weekNumber','k1']
dfr2.columns = ['userID','weekNumber','k2']
dfr3.columns = ['userID','weekNumber','k3']
dfr4.columns = ['userID','weekNumber','k4']

dfr12 = pd.merge(dfr1,dfr2, on = ['userID','weekNumber'],how='outer')
dfr123 = pd.merge(dfr12,dfr3, on = ['userID','weekNumber'],how='outer')
dfr = pd.merge(dfr123,dfr4, on = ['userID','weekNumber'],how='outer')


dfr['k_mode'] = dfr[['k1','k2','k3','k4']].mode(axis=1).iloc[:, 0]

dfr.to_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/matrix/k_list_freq_ptRots.csv',index=False)



# %%
