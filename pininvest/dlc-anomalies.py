#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
dateparse = lambda dates: pd.to_datetime(dates)


# 3. Lets define some use-case specific UDF(User Defined Functions)
def dlc_moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution

    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

# 3. Lets define some use-case specific UDF(User Defined Functions)
def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution

    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    avgs = []

    if len(data) == 0:
        raise ValueError('a cannot be empty')

    A = 2
    B = 1
    ratio = A / (window_size + B)
    ema = data[0]


    for current in data:
        ema = (current - ema) * ratio + ema
        avgs.append(ema)
    
    return avgs

def extract_outliers(timeseries):

    q1 = timeseries.quantile(0.25)
    q3 = timeseries.quantile(0.75)
    iqr = q3 - q1
    lbound = q1 - (1.5 * iqr)
    ubound = q3 + (1.5 * iqr)

    #print("ubound ->  ",ubound)
    #print("lbound ->  ",lbound)

    return list(filter(lambda value :  value < lbound or value > ubound, timeseries))

def extract_std_outliers(timeseries):
    """ Extract Outiers using std method   """
    mean = timeseries.mean()
    std = timeseries.std()
    lbound = mean - 2 * std
    ubound = mean + 2 * std
    
    return list(filter(lambda value :  value < lbound or value > ubound, timeseries))

def get_ema_anomalies(y, window_size, sigma=1.0):

    #avg = y.rolling(window_size).mean()
    avg = list(moving_average(y, window_size))
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    dates = y.index.tolist()

    keys = []
    anomalies = []

    for index in range(len(dates)):
        day = dates[index]
        y_i = y[index]
        avg_i = avg[index]
        if (y_i > avg_i + (sigma*std)) or (y_i < avg_i - (sigma*std)):
            anomalies.append(y_i)
            keys.append(day)

    return keys, anomalies

def get_ema_anomalies_rolling_std(y, window_size, sigma=1.0):
    dates = y.index.tolist()
    #avg = y.rolling(window_size).mean()
    avg = list(moving_average(y, window_size))
    residual = y - avg
    # Calculate the variation in the distribution of the residual
        # Calculate the variation in the distribution of the residual
    testing_std = residual.rolling(window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()

    keys = []
    anomalies = []

    for index, day in enumerate(dates):        
        y_i = y[index]
        avg_i = avg[index]
        rs_i = rolling_std[index]
        if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i)):
            anomalies.append(y_i)
            keys.append(day)

    return keys, anomalies, dates, rolling_std, avg, residual

def get_dlc_anomalies(y, window_size, sigma=1.0):

    avg = dlc_moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    dates = y.index.tolist()

    keys = []
    anomalies = []

    for index in range(len(dates)):
        day = dates[index]
        y_i = y[index]
        avg_i = avg[index]
        if (y_i > avg_i + (sigma*std)) or (y_i < avg_i - (sigma*std)):
            anomalies.append(y_i)
            keys.append(day)

    return keys, anomalies

def get_dlc_anomalies_rolling_std(y, window_size, sigma=1.0):
    dates = y.index.tolist()
    avg = dlc_moving_average(y, window_size).tolist()        
    residual = y - avg
    # Calculate the variation in the distribution of the residual
        # Calculate the variation in the distribution of the residual
    testing_std = residual.rolling(window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()

    keys = []
    anomalies = []

    for index in range(len(dates)):
        day = dates[index]
        y_i = y[index]
        avg_i = avg[index]
        rs_i = rolling_std[index]
        if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i)):
            anomalies.append(y_i)
            keys.append(day)

    return keys, anomalies, dates, rolling_std, avg



#work/data/FAANG-5y.csv
#work/data/FAANG-5y.csv
#work/data/us1200-y2-fundamentals.csv
#work/data/faang-5y-fundamentals.csv
assets = pd.read_csv('/home/karim/datavariance/py/data/faang-5y-fundamentals.csv', parse_dates=['day'], index_col='day', date_parser=dateparse)


aapl = assets[assets.AssetId == 341]
quarterly = aapl.NCFO.asfreq(freq='3M', method="pad")

x_anomaly, y_anomaly = get_ema_anomalies(quarterly, 4, 3)
rx_anomaly, ry_anomaly, dates, rolling_std, avg, residual = get_ema_anomalies_rolling_std(quarterly, 4, 3)


plt.title(label="Sales")
plt.plot(aapl.NCFO, label="amzn")
plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)
plt.plot(rx_anomaly, ry_anomaly, "gX", markersize=12)
plt.plot(dates, residual , label="residual")
plt.plot(dates, rolling_std, label="rolling std")
plt.plot(dates, avg, label="avg")
plt.xlabel("Time", fontsize=20)
plt.ylabel("ncfo", fontsize=20)
plt.legend()
plt.show()