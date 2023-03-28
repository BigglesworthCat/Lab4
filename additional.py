import enum


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
