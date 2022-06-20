#%%
import pandas as pd
import spherical_kde

df = pd.read_csv('/home/mindy/Desktop/BiAffect-iOS/accelAnalyses/spherical_kde/KDE_identification/sampledKDEdensities_byTOD_bw01_2500pts.csv', index_col=False)


user_list = df['userID'].unique()


dfGrp = df.groupby(['userID','weekNumber'])
for user in user_list:
    print(user)
    target_grp = dfGrp.get_group(user)

    # p = 
    # q = 

    # # calculate (P || Q)
    # kl_pq = spherical_kde.utils.spherical_kullback_liebler(p, q)
    # print('User: ' + str(cmp[0][0]) + ' day: ' + str(cmp[0][1]) + ' & User: ' + str(cmp[1][0]) + ' day: ' + str(cmp[1][1]))
    # print('KL(P || Q): %.3f bits' % kl_pq)
    # # calculate (Q || P)
    # kl_qp = spherical_kde.utils.spherical_kullback_liebler(q,p)
    # print('KL(Q || P): %.3f bits' % kl_qp)
    # # calculate symmetric KL
    # symKL = kl_pq + kl_qp
    # print('symmetric KL: %.3f bits' % symKL)








# %%
