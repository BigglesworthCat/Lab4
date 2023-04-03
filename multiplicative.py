from copy import deepcopy
from scipy import special
# from tabulate import tabulate as tb

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredText

from additional import remove_outliers

class PolynomialBuilder(object):
    def __init__(self, solution,**kwargs):
        assert isinstance(solution, Solve)
        b_gen = None
        self._solution = solution
        max_degree = max(solution.p) - 1
        if solution.cheb:
            self.symbol = 'T'
            self.basis = b_gen.sh_chebyshev(max_degree)
        elif solution.lezh:
            self.symbol = 'P'
            self.basis = b_gen.sh_legendre(max_degree)
        elif solution.lag:
            self.symbol = 'L'
            self.basis = b_gen.laguerre(max_degree)
        elif solution.erm:
            self.symbol = 'H'
            self.basis = b_gen.hermite(max_degree)
        self.a = solution.a.T.tolist()
        self.c = solution.c.T.tolist()
        self.minX = [X.min(axis=0).getA1() for X in solution.X_]
        self.maxX = [X.max(axis=0).getA1() for X in solution.X_]
        self.minY = solution.Y_.min(axis=0).getA1()
        self.maxY = solution.Y_.max(axis=0).getA1()
        self.X1_pred = self.restore_X(kwargs.get('X1', None))
        self.X2_pred = self.restore_X(kwargs.get('X2', None))
        self.X3_pred = self.restore_X(kwargs.get('X3', None))
        self.X4_pred = self.restore_X(kwargs.get('X4', None))

    def _form_lamb_lists(self):
        self.psi = list()
        for i in range(self._solution.Y.shape[1]):  # `i` is an index for Y
            psi_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                psi_i_j = list()
                for k in range(self._solution.deg[j]):  # `k` is an index for vector component
                    psi_i_jk = self._solution.Lamb[shift:shift + self._solution.p[j], i].getA1()
                    shift += self._solution.p[j]
                    psi_i_j.append(psi_i_jk)
                psi_i.append(psi_i_j)
            self.psi.append(psi_i)

    def _transform_to_standard(self, coeffs):
        std_coeffs = np.zeros(coeffs.shape)
        for index in range(coeffs.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            std_coeffs += coeffs[index] * cp
        return std_coeffs
    def restore_X(self, X):
        X_restored = X.copy()
        X_restored = X_restored.iloc[:self.prediction_step]
        for i in range(1, X.shape[1]):
            X_i = X.iloc[:, i].to_list()
            model = sm.tsa.AutoReg(X_i, lags=5)  # sm.tsa.ARIMA(X_i, order=(1, 0, 1))
            model = model.fit()
            prediciton = model.predict(start=self.N_02 + 1, end=self.N_02 + self.prediction_step)
            X_restored.iloc[:, i] = prediciton.tolist()
        for i in range(20, X.shape[1]):
            X_restored = remove_outliers(X_restored,i)
        return X_restored
    
    def _print_psi_i_jk(self, i, j, k):
        strings = list()
        for n in range(len(self.psi[i][j][k])):
            strings.append('{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.psi[i][j][k][n], j + 1, k + 1,
                                                                   symbol=self.symbol, deg=n))
        return ' + '.join(strings)

    def _print_phi_i_j(self, i, j):
        strings = list()
        for k in range(len(self.psi[i][j])):
            shift = sum(self._solution.deg[:j]) + k
            for n in range(len(self.psi[i][j][k])):
                strings.append('{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.a[i][shift] * self.psi[i][j][k][n],
                                                                       j + 1, k + 1, symbol=self.symbol, deg=n))
        return ' + '.join(strings)

    def _print_F_i(self, i):
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                for n in range(len(self.psi[i][j][k])):
                    strings.append('{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.c[i][j] * self.a[i][shift] *
                                                                           self.psi[i][j][k][n],
                                                                           j + 1, k + 1, symbol=self.symbol, deg=n))
        return ' + '.join(strings)

    def _print_F_i_transformed_denormed(self, i):
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                raw_coeffs = self._transform_to_standard(self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                    print(current_poly)
                    print(add_poly)
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                constant += current_poly[0]
                current_poly[0] = 0
                current_poly = np.poly1d(current_poly.coeffs, variable='(x{0}{1})'.format(j + 1, k + 1))
                strings.append(str(_Polynom(current_poly, '(x{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return ' +\n'.join(strings)

    def _print_F_i_transformed(self, i):
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self._solution.deg[:j]) + k
                current_poly = np.poly1d(self._transform_to_standard(self.c[i][j] * self.a[i][shift] *
                                                                     self.psi[i][j][k])[::-1],
                                         variable='(x{0}{1})'.format(j + 1, k + 1))
                constant += current_poly[0]
                current_poly[0] = 0
                strings.append(str(_Polynom(current_poly, '(x{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return ' +\n'.join(strings)

    def _print_F_i_as_sum_F_ij(self, i):
        strings = list()
        c = np.array(self.c)
        for j in range(3):
            strings.append(str(c[i][j]) + '*' + f'Φ{i + 1}{j + 1}(x{j + 1})')
        return ' +\n'.join(strings)

    def _print_F_i_as_sum_T_ij(self, i):
        strings = list()
        for j in range(3):
            for k in range(self._solution.deg[j]):
                strings.append(self._print_psi_i_jk(i, j, k))
        return ' +\n'.join(strings)

    def get_results(self):
        self._form_lamb_lists()
        f_as_sum_f = ['Φ{0}(x1,x2,x3)={result}\n'.format(i + 1, result=self._print_F_i_as_sum_F_ij(i))
                      for i in range(self._solution.Y.shape[1])]
        f_as_sum_t = ['Φ{0}(x1,x2,x3)={result}\n'.format(i + 1, result=self._print_F_i_as_sum_T_ij(i))
                      for i in range(self._solution.Y.shape[1])]
        f_normed = ['\nNormalized:\n']
        f_strings_transformed = ['Φ{0}={result}\n'.format(i + 1, result=self._print_F_i_transformed(i))
                                 for i in range(self._solution.Y.shape[1])]
        f_unnormed = ['\nUnnormalized:\n']
        f_strings_transformed_denormed = ['Φ{0}={result}\n'.format(i + 1, result=
        self._print_F_i_transformed_denormed(i))
                                          for i in range(self._solution.Y.shape[1])]
        return '\n'.join(
            f_as_sum_f + f_as_sum_t + f_normed + f_strings_transformed + f_unnormed + f_strings_transformed_denormed)

    def plot_graph(self, y_column_number, solver):
        y_column_number = y_column_number.value()
        Y = self._solution.Y_.T[y_column_number - 1].T
        F = self._solution.F_.T[y_column_number - 1].T
        error = solver.error[y_column_number - 1]
        anchored_text = AnchoredText(f'Невязка: {error}', loc=2)
        f, ax = plt.subplots(1, 1)
        ax.plot(np.arange(1, self._solution.n + 1), Y, label=f'Y_{y_column_number}')
        ax.plot(np.arange(1, self._solution.n + 1), F, label=f'F_{y_column_number}')
        ax.add_artist(anchored_text)
        plt.legend()
        plt.grid()
        plt.show()


class Solve:
    def __init__(self, d):
        self.n = d['samples']
        self.deg = d['dimensions']
        self.filename_input = d['input_file']
        self.filename_output = d['output_file']
        self.dict = d['output_file']
        self.p = list(map(lambda x: x + 1, d['degrees']))  # on 1 more because include 0
        self.norm_weight = d['norm_weight']
        self.minmax_weight = d['minmax_weigth']
        self.mean_weight = d['mean_weight']
        self.cheb = d['cheb']
        self.lezh = d['lezh']
        self.lag = d['lag']
        self.erm = d['erm']
        self.lamb_y = d['lamb_y']
        self.lamb_n = d['lamb_n']
        self.eps = 1E-6
        self.norm_error = 0.0
        self.error = 0.0

    def define_data(self):
        f = open(self.filename_input, 'r')
        self.datas = np.matrix([list(map(lambda x: float(x), f.readline().split())) for i in range(self.n)])
        self.degf = [sum(self.deg[:i + 1]) for i in range(len(self.deg))]

    def _minimize_equation(self, phi, b):
        m, _, _, _ = np.linalg.lstsq(phi, b, rcond=None)
        return m

    def norm_data(self):
        n, m = self.datas.shape
        vec = np.ndarray(shape=(n, m), dtype=float)
        for j in range(m):
            minv = np.min(self.datas[:, j])
            maxv = np.max(self.datas[:, j])
            for i in range(n):
                vec[i, j] = (self.datas[i, j] - minv) / (maxv - minv)
        self.data = np.matrix(vec)

    def define_norm_vectors(self):
        X1 = self.data[:, :self.degf[0]]
        X2 = self.data[:, self.degf[0]:self.degf[1]]
        X3 = self.data[:, self.degf[1]:self.degf[2]]
        # matrix of vectors i.e.X = [[X11,X12],[X21],...]
        self.X = [X1, X2, X3]
        self.mX = self.degf[2]
        self.Y = self.data[:, self.degf[2]:self.degf[3]]
        self.Y_ = self.datas[:, self.degf[2]:self.degf[3]]
        self.X_ = [self.datas[:, :self.degf[0]], self.datas[:, self.degf[0]:self.degf[1]],
                   self.datas[:, self.degf[1]:self.degf[2]]]

    def built_B(self):
        def B_average():
            return np.tile((self.Y.max(axis=1) + self.Y.min(axis=1)) / 2, (1, self.deg[3]))

        def B_scaled():
            return deepcopy(self.Y)

        def B_interval():
            return np.tile(self.Y.max(axis=1) - self.Y.min(axis=1), (1, self.deg[3]))

        if self.norm_weight:
            self.B = B_scaled()
        elif self.minmax_weight:
            self.B = B_interval()
        elif self.mean_weight:
            self.B = B_average()
        else:
            exit('B not definded')

    def poly_func(self):
        if self.cheb:
            self.poly_f = special.eval_sh_chebyt
        elif self.lezh:
            self.poly_f = special.eval_sh_legendre
        elif self.lag:
            self.poly_f = special.eval_laguerre
        elif self.erm:
            self.poly_f = special.eval_hermite

    def built_phi(self):
        def coordinate(v, deg):
            c = np.ndarray(shape=(self.n, 1), dtype=float)
            for i in range(self.n):
                c[i, 0] = self.poly_f(deg, v[i])
            return c

        def vector(vec, p):
            n, m = vec.shape
            a = np.ndarray(shape=(n, 0), dtype=float)
            for j in range(m):
                for i in range(p):
                    ch = coordinate(vec[:, j], i)
                    a = np.append(a, ch, 1)
            return a

        phi = np.ndarray(shape=(self.n, 0), dtype=float)
        for i in range(len(self.X)):
            vec = vector(self.X[i], self.p[i])
            phi = np.append(phi, vec, 1)
        self.phi = np.matrix(phi)

    def coef_lamb(self):
        lamb = np.ndarray(shape=(self.phi.shape[1], 0), dtype=float)
        for i in range(self.deg[3]):
            if self.lamb_y:
                boundary_1 = self.p[0] * self.deg[0]
                boundary_2 = self.p[1] * self.deg[1] + boundary_1
                lamb1 = self._minimize_equation(self.phi[:, :boundary_1], self.B[:, i])
                lamb2 = self._minimize_equation(self.phi[:, boundary_1:boundary_2], self.B[:, i])
                lamb3 = self._minimize_equation(self.phi[:, boundary_2:], self.B[:, i])
                lamb = np.append(lamb, np.concatenate((lamb1, lamb2, lamb3)), axis=1)
            else:
                lamb = np.append(lamb, self._minimize_equation(self.phi, self.B[:, i]), axis=1)
        self.Lamb = np.matrix(lamb)

    def psi(self):
        def built_psi(lamb):
            psi = np.ndarray(shape=(self.n, self.mX), dtype=float)
            q = 0
            l = 0
            for k in range(len(self.X)):
                for s in range(self.X[k].shape[1]):
                    for i in range(self.X[k].shape[0]):
                        psi[i, l] = self.phi[i, q:q + self.p[k]] * lamb[q:q + self.p[k], 0]
                    q += self.p[k]
                    l += 1
            return np.matrix(psi)

        self.Psi = []
        for i in range(self.deg[3]):
            self.Psi.append(built_psi(self.Lamb[:, i]))

    def coef_a(self):
        self.a = np.ndarray(shape=(self.mX, 0), dtype=float)
        for i in range(self.deg[3]):
            a1 = self._minimize_equation(self.Psi[i][:, :self.degf[0]], self.Y[:, i])
            a2 = self._minimize_equation(self.Psi[i][:, self.degf[0]:self.degf[1]], self.Y[:, i])
            a3 = self._minimize_equation(self.Psi[i][:, self.degf[1]:], self.Y[:, i])
            self.a = np.append(self.a, np.vstack((a1, a2, a3)), axis=1)

    def built_F1i(self, psi, a):
        m = len(self.X)  # m  = 3
        F1i = np.ndarray(shape=(self.n, m), dtype=float)
        k = 0
        for j in range(m):  # 0 - 2
            for i in range(self.n):  # 0 - 49
                F1i[i, j] = psi[i, k:self.degf[j]] * a[k:self.degf[j], 0]
            k = self.degf[j]
        return np.matrix(F1i)

    def built_Fi(self):
        self.Fi = []
        for i in range(self.deg[3]):
            self.Fi.append(self.built_F1i(self.Psi[i], self.a[:, i]))

    def coef_c(self):
        self.c = np.ndarray(shape=(len(self.X), 0), dtype=float)
        for i in range(self.deg[3]):
            m, _, _, _ = np.linalg.lstsq(self.Fi[i], self.Y[:, i], rcond=None)
            self.c = np.append(self.c, m, axis=1)

    def built_F(self):
        F = np.ndarray(self.Y.shape, dtype=float)
        for j in range(F.shape[1]):  # 2
            for i in range(F.shape[0]):  # 50
                F[i, j] = self.Fi[j][i, :] * self.c[:, j]
        self.F = np.matrix(F)
        self.norm_error = []
        for i in range(self.Y.shape[1]):
            self.norm_error.append(np.linalg.norm(self.Y[:, i] - self.F[:, i], np.inf))

    def built_F_(self):
        minY = self.Y_.min(axis=0)
        maxY = self.Y_.max(axis=0)
        self.F_ = np.multiply(self.F, maxY - minY) + minY
        self.error = []
        for i in range(self.Y_.shape[1]):
            self.error.append(np.linalg.norm(self.Y_[:, i] - self.F_[:, i], np.inf))

    def show(self):
        text = []
        text.append('\nmatrix λ:')
        text.append(tb(np.array(self.Lamb)))
        for j in range(len(self.Psi)):
            s = '\nmatrix Ψ%i:' % (j + 1)
            text.append(s)
            text.append(tb(np.array(self.Psi[j])))
        text.append('\nmatrix a:')
        text.append(tb(self.a.tolist()))
        for j in range(len(self.Fi)):
            s = '\nmatrix Φ%i:' % (j + 1)
            text.append(s)
            text.append(tb(np.array(self.Fi[j])))
        text.append('\nmatrix c:')
        text.append(tb(np.array(self.c)))
        text.append('\nError normalised (Y - Φ)')
        text.append(tb([self.norm_error]))
        text.append('\nError (Y - Φ))')
        text.append(tb([self.error]))
        return '\n'.join(text)

    def prepare(self):
        self.define_data()
        self.norm_data()
        self.define_norm_vectors()
        self.built_B()
        self.poly_func()
        self.built_phi()
        self.coef_lamb()
        self.psi()
        self.coef_a()
        self.built_Fi()
        self.coef_c()
        self.built_F()
        self.built_F_()
        self.save_to_file()
        return self.error

def _Polynom(current_poly, param):
    pass