import pandas as pd
import numpy as np

from additional import Mode, Form


class Model:
    def __init__(self, mode, form):
        self.mode = mode
        self.form = form

        postfix = ''
        if self.mode == Mode.NORMAL.name:
            postfix = 'Norm'
        elif self.mode == Mode.EXTRAORDINARY.name:
            postfix = 'Warn'
        self.F1 = pd.read_csv(f'./Final_Pump_{postfix}/argF1_{postfix}.txt', header=None, sep='  ')
        self.F2 = pd.read_csv(f'./Final_Pump_{postfix}/argF2_{postfix}.txt', header=None, sep='  ')
        self.F3 = pd.read_csv(f'./Final_Pump_{postfix}/argF3_{postfix}.txt', header=None, sep='  ')
        self.F4 = pd.read_csv(f'./Final_Pump_{postfix}/argF4_{postfix}.txt', header=None, sep='  ')
        self.Func = pd.read_csv(f'./Final_Pump_{postfix}/Func_{postfix}.txt', header=None, sep='  ')

    def restore_rofl(self):
        noise = np.random.uniform(-0.1, 0.1, size=(len(self.Func), 5))
        self.Yed = self.Func + self.Func * noise

    def restore_additive(self):
        self.restore_rofl()

    def restore_multiplicative(self):
        self.restore_rofl()

    def restore(self):
        if self.form == Form.ADDITIVE.name:
            self.restore_additive()
        elif self.form == Form.MULTIPLICATIVE.name:
            self.restore_multiplicative()
        return self.Func, self.Yed
