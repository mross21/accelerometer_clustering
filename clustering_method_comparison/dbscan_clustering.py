import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from accelerometer_utils import accel_filter, add_spher_coords, numerical_sort
from config import PATH_ACCELEROMETER_PARQUET, PATH_CLUSTERS, PATH_FIGURES
from io_utils import out_path

pd.options.mode.chained_assignment = None

def main():
    all_files = sorted(glob.glob(PATH_ACCELEROMETER_PARQUET + "*.parquet"), key = numerical_sort)

    for file in all_files:
        df = pd.read_parquet(file, engine='pyarrow')
        user = df['userID'].iloc[0]
        print(user)

        df = accel_filter(df)
        add_spher_coords(df)

        accOut = []
        dfAccByWk = df.groupby(['userID', 'weekNumber'])
        for userAndWk, accGrp in dfAccByWk:
            wk = userAndWk[1]
            print('user: ' + str(user))
            print('week: ' + str(wk))

            accGrp['x'] = pd.to_numeric(accGrp['x'])
            accGrp['y'] = pd.to_numeric(accGrp['y'])
            accGrp['z'] = pd.to_numeric(accGrp['z'])

            dbscan = DBSCAN(metric='cosine', n_jobs=-1)
            accGrp['cluster']=dbscan.fit_predict(accGrp[['x','y','z']])

            plt.rcParams.update({'font.size': 32})
            fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
            ax = fig.add_subplot()
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.scatter(accGrp['x'], accGrp['z'], c=accGrp['cluster'], cmap='Set2')
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.savefig(out_path(PATH_FIGURES, 'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_dbscan-xz.png'))

            plt.close('all')
            accOut.append(accGrp)

        if len(accOut) < 1:
            continue
        dfAccOut = pd.concat(accOut,axis=0,ignore_index=True)
        dfAccOut.to_parquet(out_path(PATH_CLUSTERS, 'user_'+str(int(user))+'_accel_withClusters-dbscan.parquet'))

    print('finish')


if __name__ == '__main__':
    main()
