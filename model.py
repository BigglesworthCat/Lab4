import statsmodels.api as sm

from additional import Mode
from methods import *
from method import *


class Model:
    def __init__(self, mode, form, N_02, prediction_step, polynomials, P_dims, weights, lambdas):
        self.mode = mode
        self.form = form

        self.N_02 = N_02
        self.prediction_step = prediction_step

        self.polynomials = polynomials
        self.P_dims = P_dims
        self.weights = weights
        self.lambdas = lambdas

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

    def restore(self,):
        kwargs = {'method' : self.form, 'prediction_step': self.prediction_step, 'N_02': self.N_02, 'Y':self.Y, 'X1': self.X1, 'X2': self.X2, 'X3': self.X3, 'X4': self.X4, 'p_dims': self.P_dims, 'polynomials' : self.polynomials }
        self.Y, self.Y_pred = restore(**kwargs)
        return self.Y, self.Y_pred

    def restore_X(self, X):
        X_restored = X.copy()
        X_restored = X_restored.iloc[:self.prediction_step]
        for i in range(1, X.shape[1]):
            X_i = X.iloc[:, i].to_list()
            model = sm.tsa.AutoReg(X_i, lags=5)  # sm.tsa.ARIMA(X_i, order=(1, 0, 1))
            model = model.fit()
            prediciton = model.predict(start=self.N_02 + 1, end=self.N_02 + self.prediction_step)
            X_restored.iloc[:, i] = prediciton.tolist()
        return X_restored
