import numpy as np
import pandas as pd

from additional import Form


def restore(**kwargs):
    method = kwargs.pop('method', None)
    Y_local = kwargs.pop('Y', None)
    prediction_step = kwargs.pop('prediction_step', None)
    
    if method == Form.ADDITIVE.name:
        return restore_additive(Y_local, prediction_step=prediction_step)
    else:
        return restore_multiplicative(Y_local, prediction_step=prediction_step)


def restore_additive(Y, prediction_step):
    Y_local = pd.concat([Y, Y[-prediction_step:].iloc[::-1]])
    noise = np.random.uniform(0, 0.04, size=(len(Y_local), 5))
    sin_t = np.tile(np.cos([i for i in range(0, len(Y_local))]), (5, 1)).T
    noise = sin_t * noise
    print(Y_local.shape)
    return Y, Y_local + Y_local * noise


def restore_multiplicative(Y, prediction_step):
    Y_local = pd.concat([Y, Y[-prediction_step:].iloc[::-1]])
    noise = np.random.uniform(0, 0.01, size=(len(Y_local), 5))
    sin_t = np.tile(np.cos([i for i in range(0, len(Y_local))]), (5, 1)).T
    noise = sin_t * noise
    print(Y_local.shape)
    return Y, Y_local + Y_local * noise
