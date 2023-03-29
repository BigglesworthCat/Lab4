import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

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
        predicted_values = []
        for i in range(0, len(self.Y) - self.N_02 + 1, self.prediction_step):
            X1 = self.X1[i:i + self.N_02]
            X2 = self.X2[i:i + self.N_02]
            X3 = self.X3[i:i + self.N_02]
            X4 = self.X4[i:i + self.N_02]

