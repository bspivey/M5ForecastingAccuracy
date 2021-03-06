#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class to predict unit sales from M5 Forecasting Accuracy data"""

from abc import abstractmethod
from re import U
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from abc import ABC, abstractmethod

__author__ = 'Ben Spivey'
__license__ = 'GNU GPLv3'
__version__ = '0.0.1'
__status__ = 'Development'

class UnitSalesFeatures:
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

    def day_string_to_int(self, df, column):
        df_copy = df.copy(deep=True)
        df_copy[column] = df_copy[column].str.extract('(\d+)')
        return df_copy

    def merge_tables(self, df_stvp_item, df_sell_prices_item):
        """Merge calendar, sales, and sell prices tables"""
        df_intermediate_item = self.df_calendar.merge(df_sell_prices_item,
                                                    how='left',
                                                    on='wm_yr_wk')

        cols_to_use = df_intermediate_item.columns.difference(df_stvp_item.columns).tolist()
        cols_to_use = cols_to_use + ['item_id', 'd', 'store_id']
        df_merged = df_stvp_item.merge(df_intermediate_item[cols_to_use],
                                                    how='inner',
                                                    on=['item_id', 'd', 'store_id'])

        df_merged = self.day_string_to_int(df_merged, 'd')

        return df_merged

    def split_train_test_data(self, df_merged):
        """Split df_merged into train, validation, and test data"""
        days_in_year = 365
        df_train = df_merged.iloc[:-2*days_in_year].copy(deep=True)
        df_validation = df_merged.iloc[-2*days_in_year:-days_in_year].copy(deep=True)
        df_test = df_merged.iloc[-days_in_year:].copy(deep=True)

        return df_train, df_validation, df_test

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

    def create_event_features(self, df_merged_store):
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

        return X_events

    def plot_sales(self, df_merged):
        df_merged_pivot = df_merged.pivot(index='date', columns='store_id', values='unit_sales')
        df_merged_pivot = df_merged_pivot.sort_values('date', axis=0, ascending=True)
        df_sales_rolling_avg = df_merged_pivot.rolling(3).mean()
        id_vars = ['date']
        store_ids = df_merged['store_id'].unique()
        df_sales_rolling_avg = pd.melt(df_sales_rolling_avg.reset_index(),
                                        id_vars=id_vars,
                                        value_vars=store_ids,
                                        var_name='store_id',
                                        value_name='unit_sales')

        fig = px.line(df_sales_rolling_avg,
                        x='date',
                        y='unit_sales',
                        color='store_id',
                        width=900,
                        height=300)
        fig.show()

class UnitSalesPrediction(ABC):
    def plot_predictions(self, X, y, y_pred):
        y = y.rolling(3).mean()
        list_of_tuples = list(zip(X.index, y, y_pred))
        columns = ['date', 'y', 'y_pred']
        df_wide = pd.DataFrame(list_of_tuples, columns=columns)
        value_vars = ['y', 'y_pred']
        df_tall = pd.melt(df_wide,
                            id_vars='date',
                            value_vars=value_vars,
                            var_name='y_label',
                            value_name='y_value')

        fig = px.line(df_tall,
                        x='date',
                        y='y_value',
                        color='y_label',
                        width=900,
                        height=300)
        fig.update_layout(
            yaxis_title='unit_sales')

        fig.show()

    @abstractmethod
    def fit_unit_sales_model(self):
        pass

    @abstractmethod
    def predict_unit_sales(self):
        pass

class UnitSalesPredictionSeasonal(UnitSalesPrediction):
    def fit_unit_sales_model(self, X_seasonal, y):
        """Trains a model to predict unit sales for one item and one store"""
        X = X_seasonal
        model = LinearRegression().fit(X, y)

        return model

    def predict_unit_sales(self, model, X_seasonal):
        X = X_seasonal.fillna(0.0)
        y_pred = pd.Series(model.predict(X),
                    index=X.index,
                    name='Predicted')

        return y_pred

class UnitSalesPredictionSeasonalAndEvent(UnitSalesPrediction):
    def fit_unit_sales_model(self, X_seasonal, X_events, y):
        """Trains a model to predict unit sales for one item and one store"""
        X = X_seasonal.join(X_events, on='date').fillna(0.0)
        model = LinearRegression().fit(X, y)

        return model

    def predict_unit_sales(self, model, X_seasonal, X_events):
        X = X_seasonal.join(X_events, on='date').fillna(0.0)
        y_pred = pd.Series(model.predict(X),
                    index=X.index,
                    name='Predicted')
        
        return y_pred

if __name__ == '__main__':
    usf = UnitSalesFeatures()
    usf.read_data()

    item_name = 'FOODS_3_069'
    store_id = 'TX_1'
    df_stvp_item, df_sell_prices_item = usf.filter_by_item(item_name)
    df_merged = usf.merge_tables(df_stvp_item, df_sell_prices_item)
    df_merged_store = df_merged[df_merged['store_id']==store_id]

    # Create training, validation, and test data
    X_seasonal, y = usf.create_seasonal_features(df_merged_store)
    X_events = usf.create_event_features(df_merged_store)
    X_s_train, X_s_validation, X_s_test = usf.split_train_test_data(X_seasonal)
    y_train, y_validation, y_test = usf.split_train_test_data(y)
    X_e_train, X_e_validation, X_e_test = usf.split_train_test_data(X_events)

    # Make training data plot
    usf.plot_sales(df_merged_store)

    # Train the model
    usp_s = UnitSalesPredictionSeasonal()
    usp_se = UnitSalesPredictionSeasonalAndEvent()

    model_s = usp_s.fit_unit_sales_model(X_s_train, y_train)
    model_se = usp_se.fit_unit_sales_model(X_s_train, X_e_train, y_train)    

    # Make training predictions
    y_pred_se = usp_se.predict_unit_sales(model_se, X_s_train, X_e_train)
    usp_se.plot_predictions(X_s_train, y_train, y_pred_se)

    # Make validation predictions
    y_pred_s = usp_s.predict_unit_sales(model_s, X_s_validation)
    usp_s.plot_predictions(X_s_validation, y_validation, y_pred_s)
    y_pred_se = usp_se.predict_unit_sales(model_se, X_s_validation, X_e_validation)
    usp_se.plot_predictions(X_s_validation, y_validation, y_pred_se)

    # Make test predictions
    y_pred_s = usp_s.predict_unit_sales(model_s, X_s_test)
    usp_s.plot_predictions(X_s_test, y_test, y_pred_s)
    y_pred_se = usp_se.predict_unit_sales(model_se, X_s_test, X_e_test)
    usp_se.plot_predictions(X_s_test, y_test, y_pred_se)