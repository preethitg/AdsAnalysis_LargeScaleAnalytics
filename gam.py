
import numpy as np
import os
import pandas as pd
import dateutil.parser
from pygam import LinearGAM

def predict_gam(ad_group,date):
    ads_file = 'data/ad_table.csv'
    df = pd.read_csv(ads_file, header=0, sep=',')
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    splines=[5, 7, 10, 20, 30, 40, 45]
    lams = np.logspace(-3,3,7)
    if(ad_group in df['ad'].unique()):
        df_ad_group_train = df[df['ad'] == ad_group]
        df_ad_group_train = df_ad_group_train.reset_index()
        df_ad_group_train['time_period'] = (df_ad_group_train['date'] - df_ad_group_train['date'][0]).dt.days
        X_train = df_ad_group_train[['time_period']].values
        y_train = df_ad_group_train['shown'].values
        #auto tuning
        gam = LinearGAM().gridsearch(X_train, y_train, lam=lams, n_splines=splines)
        predictions = gam.predict(X_train)
        print('==== Tuning for ad group %s - best generalized cross-validation %f ' % (ad_group, gam.statistics_['GCV']))
        tuning_result = (gam.lam[0][0], gam.n_splines[0], gam.statistics_['GCV'])
        predict_date = (pd.to_datetime(date) - df_ad_group_train['date'][0]).days
        print("Auto tuning result=",tuning_result)
        print("Prediction for number of ads Shown for",ad_group,"on ",date,"=",gam.predict([[predict_date]]))
        print("Regression/Lambda value = ",gam.lam)
        print("n_splines=",gam.n_splines)
    else:
        print("Ad group does not exist")
