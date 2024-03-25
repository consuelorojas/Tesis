import sys
sys.path.append('./data')
sys.path.append('./Librerias')

import errores as er
import utils_2 as ut2

# for NN

class testNN():
    def __init__(self, model, x_test, y_test, steps):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.steps = steps
        self.rolling_window_prediction = ut2.rollingWindowPrediction(self.model, self.x_test, self.steps)

    def error_metrics(self):
        mse, mape, r2, rmse = er.error_metrics(self.rolling_window_prediction[:-self.steps-1], self.y_test, self.steps)
        return mse, mape, r2, rmse
    
    def get_prediction(self):
        return self.rolling_window_prediction
    

class testSVR():
    def __init__(self, model, x_test, y_test, steps):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.steps = steps
        self.rolling_window_prediction = ut2.rollingWindowPrediction_SVR(self.model, self.x_test, self.steps)

    def error_metrics(self):
        mse, mape, r2, rmse = er.error_metrics(self.rolling_window_prediction, self.y_test, self.steps)
        return mse, mape, r2, rmse

    def get_prediction(self):
        return self.rolling_window_prediction
    