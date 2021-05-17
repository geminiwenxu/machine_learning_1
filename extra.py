import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def func_x(x, C=21.994, X=80, S0=100, m=0.4, T=0.25, sigma=0.2):
    return (np.log((C * x + C + X) / S0) - m * T) / (sigma * np.sqrt(T))


if __name__ == '__main__':
    x = np.linspace(-4, 4, 1000)
    norm_cdf = stats.norm.cdf(x)
    for idx, val in enumerate(x):
        if val < -1:
            norm_cdf[idx] = 0

    plt.plot(x, norm_cdf)
    plt.show()

