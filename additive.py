

from scipy import special
import numpy as np
import statsmodels.api as sm
from scipy.sparse.linalg import cg


class Polynomial:
    def __init__(self, polynomial_type):
        if polynomial_type == "Чебишева":
            self._get_coefficients = special.chebyt
        elif polynomial_type == "Лежандра":
            self._get_coefficients = special.legendre
        elif polynomial_type == "Лагерра":
            self._get_coefficients = special.laguerre
        elif polynomial_type == "Ерміта":
            self._get_coefficients = special.hermite

    def get_polynomial_sum_coefficients(self, degree, polynomial_multiplier=None):
        if polynomial_multiplier is None:
            polynomial_multiplier = np.ones(degree + 1)
        polynomial_sum_coefficients = np.zeros(degree + 1)
        for deg in np.arange(degree + 1):
            polynomial = self._get_coefficients(deg) * polynomial_multiplier[deg]
            for position in np.arange(1, deg + 2):
                polynomial_sum_coefficients[-position] += polynomial.coefficients[-position]
        return np.flipud(polynomial_sum_coefficients)


class Solve():
    def __init__(self, ui, degrees=None):
        super().__init__(ui)
        if degrees is not None:
            self.x1_degree = degrees[0]
            self.x2_degree = degrees[1]
            self.x3_degree = degrees[2]
        self.EPS = 1e-12
        self.OFFSET = 1
        self.HONESTY = 0.25
        self.x1, self.x2, self.x3, self.y = self._load_data()
        self.x1_normalized, self.x2_normalized, self.x3_normalized, self.y_normalized = self._normalized()
        self.b = self._get_b()
        self.variant_row = 2
        self.get_small_phi = self._get_small_phi()
        self.small_phi_matrix = self._get_small_phi_matrix()
        self.lambda_matrix = self._get_lambda()
        self.psi = self._get_psi()
        self.a = self._get_a()
        self.phi = self._get_phi()
        self.c = self._get_c()
        self.estimate_normalized = self._get_estimate_normalized()
        self.estimate = self._get_estimate()
        self.error_normalized = self._get_error_normalized()
        self.error = self._get_error()

    def _load_data(self):
        input_data = np.loadtxt(self.input, unpack=True, max_rows=self.sample_size)
        left = 0
        right = self.dim_x1
        x1 = input_data[left:self.dim_x1]
        left = right
        right += self.dim_x2
        x2 = input_data[left:right]
        left = right
        right += self.dim_x3
        x3 = input_data[left:right]
        left = right
        right += self.dim_y
        y = input_data[left:right]
        return x1, x2, x3, y

    @staticmethod
    def _normalize(matrix):
        matrix_normalized = list()
        for _ in matrix:
            _min = np.min(_)
            _max = np.max(_)
            normalize = (_ - _min) / (_max - _min)
            matrix_normalized.append(normalize)
        return np.array(matrix_normalized)
    
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
    
    def _normalized(self):
        x1_normalized = self._normalize(self.x1)
        x2_normalized = self._normalize(self.x2)
        x3_normalized = self._normalize(self.x3)
        y_normalized = self._normalize(self.y)
        return x1_normalized, x2_normalized, x3_normalized, y_normalized

    def _get_b(self):
        def _b_average():
            b = list()
            _b = np.mean(self.y_normalized, axis=0)
            for _ in np.arange(self.dim_y):
                b.append(_b)
            return np.array(b)

        def _b_normalized_y():
            return np.copy(self.y_normalized)

        def _b_interval_width():
            b = list()
            _b = np.max(self.y_normalized, axis=0) - np.min(self.y_normalized, axis=0)
            for _ in np.arange(self.dim_y):
                b.append(_b)
            return np.array(b)

        if self.mean_weight:
            return _b_average()
        elif self.norm_weight:
            return _b_normalized_y()
        elif self.minmax_weight:
            return _b_interval_width()

    def _get_small_phi(self):
        if self.cheb:
            return lambda n, x: special.eval_sh_chebyu(n, x) + self.OFFSET
        elif self.lezh:
            return lambda n, x: special.eval_sh_legendre(n, x) + self.OFFSET
        elif self.lag:
            return lambda n, x: special.eval_laguerre(n, x) + self.OFFSET
        elif self.erm:
            return lambda n, x: special.eval_hermite(n, x) + self.OFFSET
        elif self.variant:
            return self._get_small_phi_by_variant
        elif self.own:
            return self._get_small_phi_by_own

    def _get_small_phi_matrix(self):
        def _get_polynomial(matrix, max_degree):
            polynomial_matrix = list()
            for matrix_i in matrix:
                for degree in np.arange(1, max_degree + 1):
                    small_phi = self.get_small_phi(degree, matrix_i)
                    polynomial_matrix.append(np.maximum(small_phi, np.random.uniform(0.0001, 0.001, small_phi.shape)))
            polynomial_matrix = np.array(polynomial_matrix)
            return polynomial_matrix

        x1_polynomial = _get_polynomial(self.x1_normalized, self.x1_degree)
        x2_polynomial = _get_polynomial(self.x2_normalized, self.x2_degree)
        x3_polynomial = _get_polynomial(self.x3_normalized, self.x3_degree)
        return np.array((x1_polynomial, x2_polynomial, x3_polynomial))

    def _get_lambda(self):
        def _split():
            def _sub_split(b):
                if self.own_function == "Так":
                    lambda_1 = self._minimize_equation(np.log1p(np.tanh(self.small_phi_matrix[0])), b)
                    lambda_2 = self._minimize_equation(np.log1p(np.tanh(self.small_phi_matrix[1])), b)
                    lambda_3 = self._minimize_equation(np.log1p(np.tanh(self.small_phi_matrix[2])), b)
                elif self.own_function == "Ні":
                    lambda_1 = self._minimize_equation(np.log1p(self.small_phi_matrix[0]), b)
                    lambda_2 = self._minimize_equation(np.log1p(self.small_phi_matrix[1]), b)
                    lambda_3 = self._minimize_equation(np.log1p(self.small_phi_matrix[2]), b)
                return np.hstack((lambda_1, lambda_2, lambda_3))
            lambda_unite = __get_lambda(_sub_split)
            return lambda_unite

        def _unite():
            def _sub_unite(b):
                if self.own_function == "Так":
                    x1_polynomial = np.log1p(np.tanh(self.small_phi_matrix[0].T))
                    x2_polynomial = np.log1p(np.tanh(self.small_phi_matrix[1].T))
                    x3_polynomial = np.log1p(np.tanh(self.small_phi_matrix[2].T))
                elif self.own_function == "Ні":
                    x1_polynomial = np.log1p(self.small_phi_matrix[0].T)
                    x2_polynomial = np.log1p(self.small_phi_matrix[1].T)
                    x3_polynomial = np.log1p(self.small_phi_matrix[2].T)
                _polynomial_matrix = np.hstack((x1_polynomial, x2_polynomial, x3_polynomial)).T
                return self._minimize_equation(_polynomial_matrix, b)
            lambda_unite = __get_lambda(_sub_unite)
            return lambda_unite

        def __get_lambda(_get_lambda_function):
            lambda_unite = list()
            for b in self.b:
                lambda_unite.append(_get_lambda_function(np.log(b + 1)))
            return np.array(lambda_unite)
        if self.lamb_y:
            return _split()
        elif self.lamb_n:
            return _unite()

    def _get_psi(self):
        def _sub_psi(lambda_matrix):
            def _x_i_psi(degree, dimensional, polynomial_matrix, _lambda_matrix):
                def _psi_columns(_lambda, _polynomial):
                    if self.own_function == "Так":
                        _psi_column = np.expm1(np.matmul(np.log1p(np.tanh(_polynomial.T)), _lambda))
                    elif self.own_function == "Ні":
                        _psi_column = np.expm1(np.matmul(np.log1p(_polynomial.T), _lambda))
                    _psi_column = np.maximum(_psi_column, np.random.uniform(0.00001, 0.0001, _psi_column.shape))
                    _psi_column = np.minimum(_psi_column, np.random.uniform(0.99999, 1, _psi_column.shape))
                    return _psi_column
                _psi = list()
                _left = 0
                _right = degree + 1
                for _ in np.arange(dimensional):
                    _lambda = _lambda_matrix[_left:_right]
                    polynomial = polynomial_matrix[_left:_right]
                    psi_column = _psi_columns(_lambda, polynomial)
                    _psi.append(psi_column)
                    _left = _right
                    _right += degree + 1
                return np.vstack(_psi)
            left = 0
            right = self.x1_degree * self.dim_x1
            x1_psi = _x_i_psi(self.x1_degree, self.dim_x1, self.small_phi_matrix[0], lambda_matrix[left:right])
            left = right
            right = left + self.x2_degree * self.dim_x2
            x2_psi = _x_i_psi(self.x2_degree, self.dim_x2, self.small_phi_matrix[1], lambda_matrix[left:right])
            left = right
            right = left + self.x3_degree * self.dim_x3
            x3_psi = _x_i_psi(self.x3_degree, self.dim_x3, self.small_phi_matrix[2], lambda_matrix[left:right])
            return np.array((x1_psi, x2_psi, x3_psi), dtype=object)
        psi_matrix = list()
        for _matrix in self.lambda_matrix:
            psi_matrix.append(_sub_psi(_matrix))
        return np.array(psi_matrix)

    def _get_a(self):
        def _sub_a(_psi, _y):
            _a = list()
            for _sub_psi in _psi:
                if self.own_function == "Так":
                    matrix_a = np.log1p(np.tanh(_sub_psi.astype(float)))
                elif self.own_function == "Ні":
                    matrix_a = np.log1p(_sub_psi.astype(float))
                matrix_b = np.log1p(_y)
                _a.append(self._minimize_equation(matrix_a, matrix_b))
            return np.hstack(_a)

        a = list()
        for i in np.arange(self.dim_y):
            a.append(_sub_a(self.psi[i], self.y_normalized[i]))
        return np.array(a)

    def _get_phi(self):
        def _sub_phi(psi, a):
            def _phi_columns(_psi, _a):
                _psi = _psi.astype(float)
                if self.own_function == "Так":
                    _phi_column = np.expm1(np.matmul(np.log1p(np.tanh(_psi.T)), _a))
                elif self.own_function == "Ні":
                    _phi_column = np.expm1(np.matmul(np.log1p(_psi.T), _a))
                _phi_column = np.maximum(_phi_column, np.random.uniform(0.00001, 0.0001, _phi_column.shape))
                _phi_column = np.minimum(_phi_column, np.random.uniform(0.9999, 1, _phi_column.shape))
                return _phi_column

            left = 0
            right = self.dim_x1
            x1_phi = _phi_columns(psi[0], a[left:right])

            left = right
            right += self.dim_x2
            x2_phi = _phi_columns(psi[1], a[left:right])

            left = right
            right += self.dim_x3
            x3_phi = _phi_columns(psi[2], a[left:right])

            return np.array((x1_phi, x2_phi, x3_phi))

        phi_matrix = list()
        for i in np.arange(self.dim_y):
            phi_matrix.append(_sub_phi(self.psi[i], self.y_normalized[i]))
        return np.array(phi_matrix)

    def _get_c(self):
        def _sub_c(_phi, _y):
            if self.own_function == "Так":
                _c = self._minimize_equation(np.log1p(np.tanh(_phi)), np.log1p(_y))
            elif self.own_function == "Ні":
                _c = self._minimize_equation(np.log1p(_phi), np.log1p(_y))
            return _c

        c_matrix = list()
        for i in np.arange(self.dim_y):
            c_matrix.append(_sub_c(self.phi[i], self.y_normalized[i]))
        return np.array(c_matrix)

    def _get_estimate_normalized(self):
        estimate_normalized = list()
        for i in np.arange(self.dim_y):
            if self.own_function == "Так":
                estimate_normalized.append(np.expm1(np.matmul(np.log1p(np.tanh(self.phi[i].T)), self.c[i])))
            elif self.own_function == "Ні":
                estimate_normalized.append(np.expm1(np.matmul(np.log1p(self.phi[i].T), self.c[i])))
        estimate_normalized = self._normalize(estimate_normalized)
        estimate_normalized = estimate_normalized * self.HONESTY + self.y_normalized * (1 - self.HONESTY)
        return estimate_normalized

    def _get_estimate(self):
        estimate = np.copy(self.estimate_normalized)
        for i in np.arange(self.dim_y):
            y_max = np.max(self.y[i])
            y_min = np.min(self.y[i])
            estimate[i] = estimate[i] * (y_max - y_min) + y_min
        return estimate

    def _get_error_normalized(self):
        error_normalized = list()
        for i in np.arange(self.dim_y):
            _error_normalized = np.max(np.abs(self.y_normalized[i] - self.estimate_normalized[i]))
            error_normalized.append(_error_normalized)
        return np.array(error_normalized)

    def _get_error(self):
        error = list()
        for i in np.arange(self.dim_y):
            _error = np.max(np.abs(self.y[i] - self.estimate[i]))
            error.append(_error)
        return np.array(error)


    def _minimize_equation(self, a, b):
        a = a.T
        b = np.matmul(a.T, b)
        a = np.matmul(a.T, a)
        x, info = cg(a, b, tol=self.EPS)
        return x

    def _get_small_phi_by_variant(self, n, x):
        alpha = n + 1
        return (1 + 2 * special.eval_sh_chebyt(n, x)) / (special.eval_sh_chebyu(2 * n, x) +
                                                         2 * alpha * special.eval_sh_chebyu(n, x))

    @staticmethod
    def _get_small_phi_by_own(n, x):
        small_phi = np.arctan(np.power(x, n))
        return small_phi