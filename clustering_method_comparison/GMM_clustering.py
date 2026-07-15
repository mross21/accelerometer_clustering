import pandas as pd
import numpy as np
import glob
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from accelerometer_utils import accel_filter, add_spher_coords, numerical_sort
from config import PATH_ACCELEROMETER_PARQUET, PATH_CLUSTERS, PATH_FIGURES, PATH_K_LIST
from io_utils import out_path

pd.options.mode.chained_assignment = None

def main():
    dfK = pd.read_parquet(PATH_K_LIST, engine='pyarrow')
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

            try:
                k = int(dfK.loc[(dfK['userID']==user) & (dfK['weekNumber']==wk)]['k'])
            except TypeError:
                continue

            accGrp['x'] = pd.to_numeric(accGrp['x'])
            accGrp['y'] = pd.to_numeric(accGrp['y'])
            accGrp['z'] = pd.to_numeric(accGrp['z'])

            gmm = GaussianMixture(n_components=k)
            gmm.fit(accGrp[['x','y','z']])
            accGrp['cluster'] = gmm.predict(accGrp[['x','y','z']])+1

            plt.rcParams.update({'font.size': 32})
            fig = plt.figure(figsize=(16,16),facecolor=(1, 1, 1))
            ax = fig.add_subplot()
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.scatter(accGrp['x'], accGrp['z'], c=accGrp['cluster'], cmap='Set2')
            plt.xlim([-1.2,1.2])
            plt.ylim([-1.2,1.2])
            plt.savefig(out_path(PATH_FIGURES, 'user_' + str(int(user)) + '_week_' + str(int(wk)) + '_plot_gmm-xz.png'))

            plt.close('all')
            accOut.append(accGrp)

        if len(accOut) < 1:
            continue
        dfAccOut = pd.concat(accOut,axis=0,ignore_index=True)
        dfAccOut.to_parquet(out_path(PATH_CLUSTERS, 'user_'+str(int(user))+'_accel_withClusters-gmm.parquet'))

    print('finish')


if __name__ == '__main__':
    main()
