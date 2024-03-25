import sys
sys.path.append('./data')
sys.path.append('./Librerias')

import utils_2 as ut2
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

def calculate_errors(rolling_window_prediction, y_test):
    mse = mean_squared_error(y_test, rolling_window_prediction)
    mape = mean_absolute_percentage_error(y_test, rolling_window_prediction)
    r2 = r2_score(y_test, rolling_window_prediction)
    rmse = root_mean_squared_error(y_test, rolling_window_prediction)
    return mse, mape, r2, rmse

def error_metrics(rolling_window_prediction, test_set, horizon):
    _ , y_test = ut2.create_sequences(test_set, 1000, horizon)
    y_test = y_test.squeeze()
    mse, mape, r2, rmse = calculate_errors(rolling_window_prediction, y_test)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Percentage Error: {mape}')
    print(f'R2 Score: {r2}')
    print(f'Root Mean Squared Error: {rmse}')

    return mse, mape, r2, rmse

