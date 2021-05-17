import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from mlxtend.evaluate import bias_variance_decomp
from scipy import interpolate
import math


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)


def generate_data(size=25):
    x = np.random.uniform(low=0.0, high=1.0, size=25)
    epsilon_n = np.random.normal(loc=0, scale=1, size=25)
    y = np.sin(2 * np.pi * x) + 0.3 * epsilon_n
    return x, y


def regularized_model(x, y, lam):
    regularized_model = make_pipeline(GaussianFeatures(20), Ridge(lam))
    regularized_model.fit(x[:, np.newaxis], y)
    xfit = np.linspace(0, 10, 1000)
    yfit = regularized_model.predict(xfit[:, np.newaxis])
    return xfit, yfit


if __name__ == '__main__':
    """Homework 1"""
    all_x = []
    all_y = []
    all_xfit = []
    all_yfit = []
    for i in range(100):
        x, y = generate_data()
        xfit, yfit = regularized_model(x, y, lam=2.6)
        all_x.append(x)
        all_y.append(y)
        all_xfit.append(xfit)
        all_yfit.append(yfit)
        plt.plot(xfit, yfit, 'lightcoral')
        plt.xlim(0, 1)
        plt.ylim(-1.5, 1.5)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('with lamda=-2.4')
        plt.legend()

    """Homework 2"""
    xfit = np.linspace(0, 10, 1000)
    y_list = []
    for i in range(100):
        x, y = generate_data()
        xfit, yfit = regularized_model(x, y, lam=2.6)
        y_list.append(yfit)
    y_average = np.average(y_list, axis=0)
    plt.plot(xfit, y_average)
    plt.xlim(0, 1)
    plt.ylim(-1.5, 1.5)
    plt.title('with lamda=2.6')
    x = np.arange(0, 1, 0.001)
    y = np.sin(2 * np.pi * x) + 0.3
    plt.plot(x, y, 'g')
    plt.show()

    """Homework 3"""

    # x_train, y_train = generate_data(size=2500)
    # x_test, y_test = generate_data(size=1000)
    # error_fs, bias_fs, var_fs, sum_fs = [], [], [], []
    # ln_lams = np.linspace(-3.5, 3, 20, endpoint=True)
    # for ln_lam in ln_lams:
    #     model = make_pipeline(GaussianFeatures(20), Ridge(alpha=ln_lam))
    #     avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(model, x_train.reshape(-1, 1), y_train,
    #                                                                 x_test.reshape(-1, 1), y_test,
    #                                                                 loss='mse', random_seed=1)
    #     error_fs.append(avg_expected_loss)
    #     bias_fs.append(avg_bias ** 2)
    #     var_fs.append(avg_var)
    #     sum_fs.append(avg_bias ** 2 + avg_var)
    # # plt.xlim(-10, 10)
    # # plt.ylim(-10, 10)
    # plt.plot(ln_lams, error_fs, 'black', label='total_error')
    # plt.plot(ln_lams, bias_fs, 'blue', label='bias')
    # plt.plot(ln_lams, var_fs, 'red', label='variance')
    # plt.plot(ln_lams, sum_fs, 'fuchsia', label='bias^2 + var')
    # plt.legend()
    # plt.show()
