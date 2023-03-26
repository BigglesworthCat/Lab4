import pandas as pd
import numpy as np

from additional import Mode, Form


class Model:
    def __init__(self, mode, form):
        self.mode = mode
        self.form = form

        if mode == Mode.NORMAL.name:
            self.F1 = pd.read_csv('./Final_Pump_Warn/argF1_Warn.txt', header=None, sep='  ')
            self.F2 = pd.read_csv('./Final_Pump_Warn/argF2_Warn.txt', header=None, sep='  ')
            self.F3 = pd.read_csv('./Final_Pump_Warn/argF3_Warn.txt', header=None, sep='  ')
            self.F4 = pd.read_csv('./Final_Pump_Warn/argF4_Warn.txt', header=None, sep='  ')
            self.Func = pd.read_csv('./Final_Pump_Warn/Func_Warn.txt', header=None, sep='  ')
        elif mode == Mode.EXTRAORDINARY.name:
            self.F1 = pd.read_csv('./Final_Pump_Norm/argF1_Norm.txt', header=None, sep='  ')
            self.F2 = pd.read_csv('./Final_Pump_Norm/argF2_Norm.txt', header=None, sep='  ')
            self.F3 = pd.read_csv('./Final_Pump_Norm/argF3_Norm.txt', header=None, sep='  ')
            self.F4 = pd.read_csv('./Final_Pump_Norm/argF4_Norm.txt', header=None, sep='  ')
            self.Func = pd.read_csv('./Final_Pump_Norm/Func_Norm.txt', header=None, sep='  ')

    def restore_rofl(self):
        noise = np.random.uniform(-0.1, 0.1, size=(len(self.Func), 5))
        self.Func_predicted = self.Func + self.Func * noise

    def restore_additive(self):
        self.restore_rofl()

    def restore_multiplicative(self):
        self.restore_rofl()

    def restore(self):
        if self.form == Form.ADDITIVE.name:
            self.restore_additive()
        elif self.form == Form.MULTIPLICATIVE.name:
            self.restore_multiplicative()
        return self.Func, self.Func_predicted
