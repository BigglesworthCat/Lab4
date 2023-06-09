import enum

PLOT_WIDTH = 600
PLOT_HEIGHT = 200


class Mode(enum.Enum):
    NORMAL = 1
    EXTRAORDINARY = 2


class Form(enum.Enum):
    ADDITIVE = 1
    MULTIPLICATIVE = 2


class Polynom(enum.Enum):
    CHEBYSHEV = 1
    LEGANDRE = 2
    LAGERR = 3


class Weight(enum.Enum):
    NORMED = 1
    MIN_MAX = 2


class Lambda(enum.Enum):
    SINGLE_SET = 1
    TRIPLE_SET = 2

class Normalization(enum.Enum):
    NORMED = 1
    UNNORMED = 2


def remove_outliers(X,index):
    if X[index] >= max(X[:index])*1.2:
        X[index] = X[index-1]
        return X, "Outlier"
    return X, "Normal" 

    