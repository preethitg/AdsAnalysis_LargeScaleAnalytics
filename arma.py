import os
import numpy as np
import pandas as pd
import dateutil.parser
import statsmodels.api as sm
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import ARMA
from datetime import datetime
from statsmodels.stats.stattools import jarque_bera
import warnings


def predict_arma(ad_group,pred_date):
    warnings.filterwarnings("ignore")
    ads_file = 'data/ad_table.csv'
    df = pd.read_csv(ads_file, header=0, sep=',')
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    best_aic = np.inf 
    best_order = None
    best_mdl = None
    max_lag = 30
    tuning_result = {}
#     list_ad_group = set(df['ad'].values)
    if(ad_group in df['ad'].unique()):
        df_ad_group_train = df[df['ad'] == ad_group]
        df_ad_group_train = df_ad_group_train.reset_index()
        df_arma_train = df_ad_group_train[['shown', 'date']]
        series_train = pd.Series(df_arma_train['shown'], index=df_arma_train.index)
        for alpha in range(5):
            for beta in range(5):
                try:
                    tmp_mdl = ARMA(series_train.values, order=(alpha, beta)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (alpha, beta)
                        best_mdl = tmp_mdl
                except: continue
        score, pvalue, _, _ = jarque_bera(best_mdl.resid)

        if pvalue < 0.10:
            print('The residuals may not be normally distributed.')
        else:
            print('The residuals seem normally distributed.')
        tuning_result = (best_aic, best_order)
        print('Ad_group: {} aic: {:6.2f} | best order: {}'.format(ad_group, best_aic, best_order))

        df_ad_group_train['time_period'] = (df_ad_group_train['date'] - df_ad_group_train['date'][0]).dt.days
        X = df_ad_group_train[['time_period']].values
        y = df_ad_group_train['shown'].values
        series_train.plot(title='Shown values trend', color = 'C1')
        plt.ylabel('shown values')
        plt.xlabel('Days gap from 2015-10-01')
        plt.scatter(X, y, facecolor='gray', edgecolors='none')
        plt.show()
        #check for auto correlation
        lag_plot(series_train)
        plt.show()
        autocorrelation_plot(series_train)
        plt.show()
        plot_acf(series_train.values, lags=max_lag)
        plt.show()

        data = series_train.values
        data = data.astype('float32')
        model = ARMA(data, order=best_order)
#         model_fit = model.fit(transparams=False)
        try:
            model_fit = model.fit(transparams=False)
            model_fit.plot_predict(plot_insample=True)
            plt.scatter(X, y, color = 'gray')
            plt.title('ARMA')
            plt.show()
            days_gap = (pd.to_datetime(pred_date) - df_arma_train['date'][0]).days
            forecast = model_fit.forecast(steps=days_gap)

            print('Prediction of shown value for',pred_date,'=')
            print(forecast[0][0])
        except ValueError:
            print('This data is not suitable for ARMA')
    else:
        print("Ad group does not exist")


