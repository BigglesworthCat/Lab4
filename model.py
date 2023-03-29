import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

from additional import Mode, Form


class Model:
    def __init__(self, mode, form, N_02, prediction_step):
        self.mode = mode
        self.form = form

        self.N_02 = N_02
        self.prediction_step = prediction_step

        postfix = ''
        if self.mode == Mode.NORMAL.name:
            postfix = 'Norm'
        elif self.mode == Mode.EXTRAORDINARY.name:
            postfix = 'Warn'
        self.X1 = pd.read_csv(f'./Final_Pump_{postfix}/argF1_{postfix}.txt', header=None, sep='  ')
        self.X2 = pd.read_csv(f'./Final_Pump_{postfix}/argF2_{postfix}.txt', header=None, sep='  ')
        self.X3 = pd.read_csv(f'./Final_Pump_{postfix}/argF3_{postfix}.txt', header=None, sep='  ')
        self.X4 = pd.read_csv(f'./Final_Pump_{postfix}/argF4_{postfix}.txt', header=None, sep='  ')
        self.Y = pd.read_csv(f'./Final_Pump_{postfix}/Func_{postfix}.txt', header=None, sep='  ')

    def restore_rofl(self):
        noise = np.random.uniform(-0.1, 0.1, size=(len(self.Y), 5))
        self.Y_pred = self.Y + self.Y * noise

    def restore_additive(self):
        self.restore_rofl()

    def restore_multiplicative(self):
        self.restore_rofl()

    def restore(self):
        if self.form == Form.ADDITIVE.name:
            self.restore_additive()
        elif self.form == Form.MULTIPLICATIVE.name:
            self.restore_multiplicative()
        return self.Y, self.Y_pred

    def exponential_smoothing(self, values, alpha):
        """
        Applies exponential smoothing to a column of a given dataframe using the specified alpha value.
        """
        smoothed_values = [values[0]] # initialize the first smoothed value with the first value in the column
        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed_values[-1]
            smoothed_values.append(smoothed_value)
        return pd.Series(smoothed_values)


    def restore_x(self,X):
        X_restored = X.copy()
        X_restored = X_restored.iloc[:self.prediction_step]
        for i in range(1, X.shape[1]):
            X_i = X.iloc[:,i].to_list()
            model = sm.tsa.AutoReg(X_i, lags=5) #sm.tsa.ARIMA(X_i, order=(1, 0, 1))
            model = model.fit()
            prediciton = model.predict(start=self.N_02+1, end=self.N_02 + self.prediction_step)
            X_restored.iloc[:,i] = prediciton.tolist()
        return X_restored
            
            
            

    def restore_y(self,X,Y):
        X_restored = self.restore_x(X).iloc[:, 1:]
        regressor = LinearRegression()
        X = X.iloc[:,1:]
        
        # preprocess data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        regressor.fit(X_scaled, Y)
        X_scaled_restored = scaler.fit_transform(X_restored)
        y_pred = regressor.predict(X_scaled_restored)
        # self.Y_pred = self.Y.copy()
        # noise = np.random.uniform(-0.1, 0.1, size=(len(self.Y), 5))
        # self.Y_pred = self.Y + self.Y * noise
        # self.Y_pred.iloc[:, 1] = y_pred
        # self.Yed.iloc[:,1] = self.exponential_smoothing(self.Yed.iloc[:,1].to_list(),0.1)
        return y_pred
        
    def restore_linear(self):
        '''
        X = self.Y1.iloc[:, 1:]
        y = self.Y.iloc[:, 1]
        regressor = RandomForestRegressor()
        # preprocess data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        regressor.fit(X_scaled, y)
        y_pred = regressor.predict(X_scaled)
        self.Y_pred = self.Y.copy()
        noise = np.random.uniform(-0.1, 0.1, size=(len(self.Y), 5))
        self.Y_pred = self.Y + self.Y * noise
        self.Y_pred.iloc[:, 1] = y_pred
        # self.Yed.iloc[:,1] = self.exponential_smoothing(self.Yed.iloc[:,1].to_list(),0.1)
        return self.Y, self.Y_pred
        '''
        self.Y_pred = self.Y.copy()
        predicted_values = []
        for i in range(0, len(self.Y) - self.N_02 + 1, self.prediction_step):
            X1 = self.X1.iloc[i:i + self.N_02]
            X2 = self.X2.iloc[i:i + self.N_02]
            X3 = self.X3.iloc[i:i + self.N_02]
            X4 = self.X4.iloc[i:i + self.N_02]
            Y1 = self.Y.iloc[i:i + self.N_02,1]
            Y2 = self.Y.iloc[i:i + self.N_02,2]
            Y3 = self.Y.iloc[i:i + self.N_02,3]
            Y4 = self.Y.iloc[i:i + self.N_02,4] 
            
            self.Y_pred.iloc[i:i + self.prediction_step,1] = self.restore_y(X1,Y1)
            self.Y_pred.iloc[i:i + self.prediction_step,2] = self.restore_y(X2,Y2)
            self.Y_pred.iloc[i:i + self.prediction_step,3] = self.restore_y(X3,Y3)
            self.Y_pred.iloc[i:i + self.prediction_step,4] = self.restore_y(X4,Y4)

        return self.Y, self.Y_pred
