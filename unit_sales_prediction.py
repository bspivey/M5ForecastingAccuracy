#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class to predict unit sales from M5 Forecasting Accuracy data"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

__author__ = 'Ben Spivey'
__license__ = 'GNU GPLv3'
__version__ = '0.0.1'
__status__ = 'Development'

class UnitSalesPrediction:
    def __init__(self):
        self.folder = ''

    def read_data(self, folder='m5-forecasting-accuracy/'):
        """Read three tables for the event calendar, item sales by day, and sell prices by week"""
        self.folder = folder
        self.df_calendar = pd.read_csv(folder + 'calendar.csv')
        self.df_sales_train_validation = pd.read_csv(folder + 'sales_train_validation.csv')
        self.df_sell_prices = pd.read_csv(folder + 'sell_prices.csv')

        print(self.df_calendar.head(5))
        print(self.df_calendar.event_name_1.unique())
        print(self.df_calendar.event_type_1.unique())
        print(self.df_calendar.event_name_2.unique())
        print(self.df_calendar.event_type_2.unique())
        print(self.df_sales_train_validation.head(5))
        print(self.df_sales_train_validation.dept_id.unique())
        print(self.df_sales_train_validation.item_id.unique())
        print(self.df_sell_prices.head(5))

    def filter_by_item(self, item_id):
        """Pivot all tables to tall and create dataframes for only one item"""
        days = ['d_' + str(i) for i in range(1, 1914)]
        id_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        df_sales_train_validation_pivot = pd.melt(self.df_sales_train_validation,
                                                    id_vars=id_vars,
                                                    value_vars=days,
                                                    var_name='d',
                                                    value_name='unit_sales')

        df_stvp_item = df_sales_train_validation_pivot[df_sales_train_validation_pivot['item_id']==item_id]
        df_sell_prices_item = self.df_sell_prices[self.df_sell_prices['item_id']==item_id]

        return df_stvp_item, df_sell_prices_item

    def merge_tables(self, df_stvp_item, df_sell_prices_item):
        """Merge calendar, sales, and sell prices tables"""
        df_intermediate_item = self.df_calendar.merge(df_sell_prices_item,
                                                    how='left',
                                                    on='wm_yr_wk')

        print('merge_tables')
        print(df_intermediate_item.head(5))
        print(df_stvp_item.head(5))
        print(df_sell_prices_item.head(5))

        cols_to_use = df_intermediate_item.columns.difference(df_stvp_item.columns).tolist()
        cols_to_use = cols_to_use + ['item_id', 'd', 'store_id']
        df_merged = df_stvp_item.merge(df_intermediate_item[cols_to_use],
                                                    how='inner',
                                                    on=['item_id', 'd', 'store_id'])

        return df_merged

    def create_seasonal_features(self, df_merged_store):
        """Creates seasonal features for one item and one store"""
        df_copy = df_merged_store.copy(deep=True)
        y = df_copy['unit_sales']

        df_copy['date'] = pd.DatetimeIndex(df_copy['date'])
        df_copy.set_index('date', inplace=True)
        fourier = CalendarFourier(freq='A', order=16)
        dp = DeterministicProcess(index=df_copy.index,
                                    constant=True,
                                    order=1,
                                    seasonal=True,
                                    additional_terms=[fourier],
                                    drop=True)
        X = dp.in_sample()

        return X, y

    def predict_unit_sales(self, df_merged_store, X, y):
        """Predicts unit sales for one item and one store"""
        df_copy = df_merged_store.copy(deep=True).fillna('')
        df_copy = df_copy[['date', 'event_name_1']]
        df_copy['date'] = pd.DatetimeIndex(df_copy['date'])
        df_copy.set_index('date', inplace=True)

        ohe = OneHotEncoder(sparse=False)
        X_events = pd.DataFrame(
            ohe.fit_transform(df_copy),
            index=df_copy.index,
            columns=df_copy['event_name_1'].unique(),
        )
        print(X.dtypes)
        print(X_events.dtypes)
        X2 = X.join(X_events, on='date').fillna(0.0)
        model = LinearRegression().fit(X2, y)
        y_pred = pd.Series(model.predict(X2),
                            index=X2.index,
                            name='Predicted')

        return y_pred

    def plot_predictions(self, X, y, y_pred):
        list_of_tuples = list(zip(X.index, y, y_pred))
        columns = ['date', 'y', 'y_pred']
        df_wide = pd.DataFrame(list_of_tuples, columns=columns)
        value_vars = ['y', 'y_pred']
        df_tall = pd.melt(df_wide, id_vars='date', value_vars=value_vars, var_name='y_label', value_name='y_value')

        print('df_tall')
        print(df_tall.head(10))

        fig = px.line(df_tall, x='date', y='y_value', color='y_label')
        fig.show()

    def plot_sales(self, df_merged):
        df_merged_pivot = df_merged.pivot(index='d', columns='store_id', values='unit_sales')
        df_sales_rolling_avg = df_merged_pivot.rolling(7).mean()
        id_vars = ['d']
        store_ids = df_merged['store_id'].unique()
        df_sales_rolling_avg = pd.melt(df_sales_rolling_avg.reset_index(), id_vars=id_vars,
                                        value_vars=store_ids, var_name='store_id', value_name='unit_sales')

        fig = px.line(df_sales_rolling_avg, x='d', y='unit_sales', color='store_id')
        fig.show()

if __name__ == '__main__':
    unit_sales_prediction = UnitSalesPrediction()
    unit_sales_prediction.read_data()

    item_name = 'FOODS_3_069'
    df_stvp_item, df_sell_prices_item = unit_sales_prediction.filter_by_item(item_name)
    df_merged = unit_sales_prediction.merge_tables(df_stvp_item, df_sell_prices_item)

    print('main')
    pd.set_option('display.max_columns', None)
    print(df_merged.head(5))

    df_merged_store = df_merged[df_merged['store_id']=='TX_1']
    X, y = unit_sales_prediction.create_seasonal_features(df_merged_store)
    y_pred = unit_sales_prediction.predict_unit_sales(df_merged_store, X, y)
    unit_sales_prediction.plot_predictions(X, y, y_pred)

    #unit_sales_prediction.plot_sales(df_merged)