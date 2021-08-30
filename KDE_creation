import pandas as pd

f = '/media/mindy/40b181d3-d7fc-4d1a-9538-3bdb4f25fa12/BiAffect-iOS/accelAnalyses/analysis/accel_df_unmasck.csv'
df = pd.read_csv(f, index_col=False)

# remove any values over 1.1
df = df[(df['medianX'] < 1.1) & (df['medianY'] < 1.1) & (df['medianZ'] < 1.1)]
