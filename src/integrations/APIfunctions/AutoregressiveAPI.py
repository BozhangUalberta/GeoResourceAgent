import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from pykalman import KalmanFilter

class autoregressive_extrapolation():
    def __init__(self):
        pass
    #%%
    def moving_average(self, data: pd.DataFrame, value_column: str, extend_length=36):
        """
        Calculate the moving average of a time series and extend it by a given length.
        
        :param data: pandas DataFrame containing the time series data.
        :param value_column: the name of the column in the DataFrame with the values.
        :param extend_length: the number of steps to extend the moving average.
        :return: extended time series as a numpy array.
        """
        values = data[value_column].values
        moving_avg = np.mean(values[-extend_length:])

        # Extend the sequence
        extended_sequence = np.concatenate([values, np.full(extend_length, moving_avg)])

        return extended_sequence
    
    #%%
    def autoregressive_MA(self, data: pd.DataFrame, value_column: str, extend_length=36, order=(1, 0)):
        """
        Apply an Autoregressive Moving Average (ARMA) model to the data and extend the forecast.
        
        :param data: pandas DataFrame containing the time series data.
        :param value_column: the name of the column in the DataFrame with the values.
        :param extend_length: the number of steps to extend the forecast.
        :param order: the (p, q) order of the ARMA model.
        :return: extended time series as a numpy array.
        """
        values = data[value_column].values

        # Fit the ARMA model
        model = ARIMA(values, order=(order[0], 0, order[1]))
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps=extend_length)
        extended_sequence = np.concatenate([values, forecast])

        return extended_sequence
    
    #%%
    def autoregressive_integrated_MA(self, data: pd.DataFrame, value_column: str, extend_length=36, order=(1, 1, 0)):
        """
        Apply an Autoregressive Integrated Moving Average (ARIMA) model to the data and extend the forecast.
        
        :param data: pandas DataFrame containing the time series data.
        :param value_column: the name of the column in the DataFrame with the values.
        :param extend_length: the number of steps to extend the forecast.
        :param order: the (p, d, q) order of the ARIMA model.
        :return: extended time series as a numpy array.
        """
        values = data[value_column].values

        # Fit the ARIMA model
        model = ARIMA(values, order=order)
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps=extend_length)
        extended_sequence = np.concatenate([values, forecast])

        return extended_sequence
    
    #%%
    def seasonal_ARIMA(self, data: pd.DataFrame, value_column: str, extend_length=36, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Apply a Seasonal ARIMA (SARIMA) model to the data and extend the forecast.
        
        :param data: pandas DataFrame containing the time series data.
        :param value_column: the name of the column in the DataFrame with the values.
        :param extend_length: the number of steps to extend the forecast.
        :param order: the (p, d, q) order of the SARIMA model.
        :param seasonal_order: the (P, D, Q, S) seasonal order of the SARIMA model.
        :return: extended time series as a numpy array.
        """
        values = data[value_column].values
        model = SARIMAX(values, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=extend_length)
        extended_sequence = np.concatenate([values, forecast])
        return extended_sequence

    #%%
    def exponential_smoothing(self, data: pd.DataFrame, value_column: str, extend_length=36, trend='add', seasonal='add', seasonal_periods=12):
        """
        Apply Exponential Smoothing (ETS) to the data and extend the forecast.
        
        :param data: pandas DataFrame containing the time series data.
        :param value_column: the name of the column in the DataFrame with the values.
        :param extend_length: the number of steps to extend the forecast.
        :param trend: type of trend component ('add', 'mul', or None).
        :param seasonal: type of seasonal component ('add', 'mul', or None).
        :param seasonal_periods: number of periods in a season.
        :return: extended time series as a numpy array.
        """
        values = data[value_column].values
        model = ExponentialSmoothing(values, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=extend_length)
        extended_sequence = np.concatenate([values, forecast])
        return extended_sequence

    #%%
    def vector_autoregression(self, data: pd.DataFrame, extend_length=36, lags=1):
        """
        Apply Vector Autoregression (VAR) to multivariate data and extend the forecast.
        
        :param data: pandas DataFrame containing the time series data for multiple variables.
        :param extend_length: the number of steps to extend the forecast.
        :param lags: the number of lag observations to include.
        :return: extended time series as a numpy array
        """
        model = VAR(data)
        model_fit = model.fit(lags)
        forecast = model_fit.forecast(data.values[-lags:], steps=extend_length)
        extended_array = np.concatenate([data.values, forecast], axis=0)
        return extended_array
    
    #%%
    def kalman_filter(self, data: pd.DataFrame, value_column: str, extend_length=36):
        """
        Apply Kalman Filters to the data and extend the forecast.
        
        :param data: pandas DataFrame containing the time series data.
        :param value_column: the name of the column in the DataFrame with the values.
        :param extend_length: the number of steps to extend the forecast.
        :return: extended time series as a numpy array.
        """
        values = data[value_column].values
        kf = KalmanFilter(initial_state_mean=values[0], n_dim_obs=1)
        kf = kf.em(values, n_iter=5)
        state_means, _ = kf.smooth(values)

        # Predict future values
        future_state_means = kf.filter_update(state_means[-1], np.eye(1), observation=None)[0]
        forecast = np.concatenate([future_state_means for _ in range(extend_length)])
        
        extended_sequence = np.concatenate([values, forecast])
        return extended_sequence
