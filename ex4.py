import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det, matrix_power
from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot(mean, cov):
    x1, x2 = np.random.multivariate_normal(mean, cov, 1000).T
    plt.hist2d(x1, x2, bins=(30, 30), cmap='Greens')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.savefig("1000 samples distributed blue bins", bbox_inches='tight')


def mle(m):
    x1, x2 = np.random.multivariate_normal(mean, cov, m).T
    # print("x1 and x2", x1, x2)
    # print("x1", x1)
    # print("the sum of x1", sum(x1))
    mu_ml_1 = (1 / m) * sum(x1)
    # print("mu_ml_1", mu_ml_1)
    mu_ml_2 = (1 / m) * sum(x2)
    # print("mu_ml_2", mu_ml_2)
    mu_ml = np.array([mu_ml_1, mu_ml_2])
    # print("mu_ml", mu_ml)

    sigma_ml_11_sum = 0
    sigma_ml_12_sum = 0
    sigma_ml_21_sum = 0
    sigma_ml_22_sum = 0
    for i in range(m):
        # print(i)
        # print(x1[i])
        sigma_ml_11_sum += (x1[i] - mu_ml_1) * (x1[i] - mu_ml_1)
        sigma_ml_12_sum += (x2[i] - mu_ml_2) * (x1[i] - mu_ml_1)
        sigma_ml_21_sum += (x1[i] - mu_ml_1) * (x2[i] - mu_ml_2)
        sigma_ml_22_sum += (x2[i] - mu_ml_2) * (x2[i] - mu_ml_2)
    # print(sigma_ml_11_sum, sigma_ml_12_sum, sigma_ml_21_sum, sigma_ml_22_sum)
    # print(sigma_ml_11_sum/m)
    sigma_ml = np.array([[sigma_ml_11_sum / m, sigma_ml_12_sum / m], [sigma_ml_21_sum / m, sigma_ml_22_sum / m]])
    # print(sigma_ml)
    return mu_ml, sigma_ml


def likelihood_function(mu, cov, data):
    covinv = inv(cov)
    # data = np.random.multivariate_normal(mu, cov, 20)
    temp3 = 0
    for i in range(4):
        temp = np.array([data[i][0] - mu[0], data[i][1] - mu[1]])
        print(data[i][0])
        print(data[i][1])
        print("temp", temp)
        temp1 = temp.T.dot(covinv)
        print("temp transpose", temp.T)
        print("covinv", covinv)
        print("temp1", temp1)
        temp2 = temp1.dot(temp)
        print("temp2", temp2)
        temp3 += temp2
        print("temp3", temp3)
    temp4 = (-0.5) * temp3
    return (2 * 3.14) ** (-4) * det(cov) ** (-2) * np.exp(temp4)


if __name__ == '__main__':
    mean = [2, 3]
    cov = [[4, -0.5], [-0.5, 2]]
    plot(mean, cov)

    print("2 data points", mle(2))
    print("20data points", mle(20))
    print("200 data points", mle(200))
    print("2000 data points", mle(2000))

    mean = [2, 3]
    cov = [[4, -0.5], [-0.5, 2]]
    point = np.random.multivariate_normal(mean, cov, 4)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    mu = np.arange(0.1, 10, 1)
    var = np.arange(0.1, 10, 1)
    X, Y = np.meshgrid(mu, var)
    Z = []
    for v in var:
        row = []
        cov = [[v, -0.5], [-0.5, 2]]
        for m in mu:
            mean = [m, 3]
            L = likelihood_function(mean, cov, point)
            # print(L)
            row.append(L)
        Z.append(row)
    Z = np.array(Z)

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", vmin=0, vmax=0.000007)
    ax.set_xlabel("mu")
    ax.set_ylabel("variance")
    ax.set_zlabel("Likelihood")
    ax.set_zlim(0, 0.000001)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()







    # x = arange(-3.0, 3.0, 0.1)
    # y = arange(-3.0, 3.0, 0.1)
    # matrixSize = 2
    # A = np.random.rand(matrixSize, matrixSize)
    # B = np.dot(A, A.transpose())
    # mu, cov = meshgrid(x, y)
    # pr = likelihood_function(mu, cov)
    # im = imshow(pr)
    # show()

# mean = [2, 3]
# cov = [[4, -0.5], [-0.5, 2]]
# covinv = inv(cov)
# data = []
# for i in range(m):
#     datapoint = np.random.multivariate_normal(mean, cov).T
#     print(datapoint)
#     data.append(datapoint)
# print("data collection", data)
# temp3 = []
# for i in range(m):
#     temp = np.array([data[i][0] - 2, data[i][1] - 3])
#     print(data[i][0])
#     print(data[i][1])
#     print(data[i][0] - 2)
#     print(data[i][1] - 3)
#     temp1 = temp.T.dot(covinv)
#     print("temp transpose", temp.T)
#     print("covinv", covinv)
#     print("temp1", temp1)
#     temp2 = temp1.dot(temp)
#     print("temp2", temp2)
#     temp3.append(temp2)
# print(temp3)
# temp4 = np.array(temp3)
# print("temp4", temp4)
# pr = []
# for i in range(m):
#     print("inside exp", -0.5 * temp4[i])
#     temp_pr = (2 * 3.14) ** (-m) * np.linalg.det(cov) ** (-m / 2) * math.exp(-0.5 * temp4[i])
#     pr.append(temp_pr)
# return pr
