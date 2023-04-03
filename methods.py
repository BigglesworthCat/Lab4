from additional import Form
from multiplicative import Solve, PolynomialBuilder
from additive import Solve as SolveMuliplicative
from additive import Polynomial as PolynomialMuliplicative

def restore_additive(**kwargs):
    s = Solve(**kwargs)
    s.prepare()
    return PolynomialBuilder(s)
    

def restore_multiplicative(**kwargs):
    s = SolveMuliplicative(**kwargs)
    s.prepare()
    return PolynomialMuliplicative(s)

def restore(**kwargs):
    f = None
    method = kwargs.pop('method', None)
    
    if method == Form.ADDITIVE.name:
        f = restore_additive
    else:
        f = restore_multiplicative

    Y = kwargs.pop('Y', None)
    N_02 = kwargs.pop('N_02', None)
    prediction_step = kwargs.pop('prediction_step', None)
    
    Y_pred = Y.copy()
    for i in range(0, len(Y) - N_02 + 1, prediction_step):
        X1 = X1.iloc[i:i + N_02]
        X2 = X2.iloc[i:i + N_02]
        X3 = X3.iloc[i:i + N_02]
        X4 = X4.iloc[i:i + N_02]
        Y1 = Y.iloc[i:i + N_02, 1]
        Y2 = Y.iloc[i:i + N_02, 2]
        Y3 = Y.iloc[i:i + N_02, 3]
        Y4 = Y.iloc[i:i + N_02, 4]

        Y_pred.iloc[i:i + prediction_step, 1] = f(X1, Y1)
        Y_pred.iloc[i:i + prediction_step, 2] = f(X2, Y2)
        Y_pred.iloc[i:i + prediction_step, 3] = f(X3, Y3)
        Y_pred.iloc[i:i + prediction_step, 4] = f(X4, Y4)
    return Y, Y_pred