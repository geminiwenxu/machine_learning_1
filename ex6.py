import math
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal


def sample(D):
    mean = np.zeros(shape=D, dtype=int)
    cov = np.identity(D)
    sample = np.random.multivariate_normal(mean, cov, size=1000)
    return sample


def euclidean_dist(D, sample):
    ls_dist = []
    mean = np.zeros(shape=D, dtype=int)
    for elem in sample:
        dist = np.linalg.norm(elem - mean)
        ls_dist.append(dist)
    return ls_dist


def density(D, sample):
    ls_density = []
    mean = np.zeros(shape=D, dtype=int)
    cov = np.identity(D)
    for elem in sample:
        density = multivariate_normal.pdf(elem, mean=mean, cov=cov)
        ls_density.append(density)
    return ls_density


def generate_data():
    X = np.random.uniform(low=0.0, high=1.0, size=25)
    epsilon_n = np.random.normal(loc=0, scale=0.3, size=25)
    y = np.sin(2 * np.pi * X) + epsilon_n
    return X, y


def generate_test_data():
    X_test = np.random.uniform(low=0.0, high=1.0, size=1000)
    epsilon_n = np.random.normal(loc=0, scale=0.3, size=1000)
    y_test = np.sin(2 * np.pi * X_test) + epsilon_n
    return X_test, y_test


def design_matrix(X):
    part_Sigma = []
    u = np.linspace(0, 1, 9)
    s = 0.1
    for j in range(9):
        for i in range(25):
            sigma = np.exp(((X[i] - u[j]) ** 2) / (-2 * s ** 2))
            part_Sigma.append(sigma)
    return part_Sigma


def ex2():
    # Lasson Path
    N = 1000
    D = 8
    X = np.random.normal(size=(N, D))
    w = np.array([-10, 5, 3, 4, 1, 0.5, 0.25, -100])
    y = np.dot(X, w) + 0.01 * np.random.normal(size=N)
    alphas, coefs, _ = lasso_path(X, y)
    return alphas, coefs

if __name__ == '__main__':
    """Homework 1"""
    # D = 10
    # sample = sample(D)
    # ls_dist = euclidean_dist(D, sample)
    # ls_density = density(D, sample)
    # fig, axs = plt.subplots(2)
    # fig.suptitle("Dimension: D={}.".format(D))
    # axs[0].hist(ls_dist, bins=100, color='pink')
    # axs[0].set_title('The distance')
    # axs[1].hist(ls_density, bins=100, color='yellowgreen')
    # axs[1].set_title('The density')
    # plt.show()

    """Homework 2"""
    # df = pd.read_csv('/Users/geminiwenxu/PycharmProjects/ml_plot/Data/training_data.csv')
    # target_col = "y"
    # X = df.loc[:, df.columns != target_col]
    # y = df.loc[:, target_col]
    # ls_alphas = np.linspace(-6, 4, 200, endpoint=True)
    # alphas, coef_path, _ = lasso_path(X.to_numpy(), y, alphas=ls_alphas)
    # print("alphas", alphas)
    # print("coef_path", coef_path)
    # for each_ceof in coef_path:
    #     print("each_ceof", each_ceof)
    #     print(len(each_ceof))
    #     plt.plot(alphas, each_ceof)
    #     plt.title('Lasso')
    #     plt.xlabel('alphas')
    #     plt.ylabel('coefficients')
    # plt.show()

    '''Correct answer of Homework 2'''
    alphas, coefs = ex2()
    plt.plot(np.log(alphas), coefs.T)
    plt.show()


    """Homework 3"""
    # X, y = generate_data()
    # alpha = np.linspace(math.exp(-5), math.exp(5), 1000)
    # ln_alpha = np.log(alpha)
    # beta = 0.3 ** (-2)
    # part_Sigma = design_matrix(X)
    # ls_part_Sigma = np.split(np.array(part_Sigma), 9)
    # Sigma = np.ones(25), ls_part_Sigma[0], ls_part_Sigma[1], ls_part_Sigma[2], ls_part_Sigma[3], ls_part_Sigma[4], \
    #         ls_part_Sigma[5], ls_part_Sigma[6], ls_part_Sigma[7], ls_part_Sigma[8]
    # eigenvalue, eigenvector = np.linalg.eig(beta * np.dot(Sigma, np.transpose(Sigma)))
    # print(eigenvalue)
    # gamma = []
    # for q in range(1000):
    #     lambda_ = 0
    #     for k in range(10):
    #         temp = eigenvalue[k] / (eigenvalue[k] + alpha[q])
    #         lambda_ += temp
    #         print(f' for eigenvalue: {eigenvalue[k]} and alpha: {alpha[q]}, we get lambda: {lambda_}')
    #     gamma.append(lambda_)
    # plt.plot(ln_alpha, gamma, color='green')
    #
    # two_alpha_E_m_N = []
    # for l in range(1000):
    #     A = alpha[l] * np.identity(10) + beta * np.dot(Sigma, np.transpose(Sigma))
    #     m_N = beta * np.dot(np.dot(np.linalg.inv(A), Sigma), y)
    #     each_Ew_m_N = 0.5 * alpha[l] * np.dot(np.transpose(m_N), m_N)
    #     result = 2 * alpha[l] * each_Ew_m_N
    #     two_alpha_E_m_N.append(result)
    # plt.plot(ln_alpha, two_alpha_E_m_N, color='blue')
    # plt.ylim(0, 100)
    # plt.show()
    #
    # M = 10
    # N = 25
    # ls_log_evidence = []
    # ls_temp = []
    # for h in range(1000):
    #     A = alpha[h] * np.identity(10) + beta * np.dot(Sigma, np.transpose(Sigma))
    #     m_N = beta * np.dot(np.dot(np.linalg.inv(A), Sigma), y)
    #     each_E_m_N = (beta / 2) * np.dot(np.transpose((y - np.dot(np.transpose(Sigma), m_N))),
    #                                      (y - np.dot(np.transpose(Sigma), m_N))) + 0.5 * alpha[h] * np.dot(
    #         np.transpose(m_N), m_N)
    #     log_evidence = (M / 2) * np.log10(alpha[h]) + (N / 2) * np.log10(beta) - each_E_m_N - (1 / 2) * np.log10(
    #         det(A)) - (N / 2) * np.log10(2 * np.pi)
    #     ls_log_evidence.append(log_evidence)
    # plt.plot(ln_alpha, ls_log_evidence, color='green')
    #
    # weight = []
    # for h in range(1000):
    #     A = alpha[h] * np.identity(10) + beta * np.dot(Sigma, np.transpose(Sigma))
    #     m_N = beta * np.dot(np.dot(np.linalg.inv(A), Sigma), y)
    #     weight.append(m_N)
    # print(np.shape(weight))
    # print(weight[0])
    #
    # ls_error = []
    # for t in range(1000):
    #     prediction_y = np.dot(np.transpose(weight[t]), Sigma)
    #     # error = sum(abs(y - prediction_y))
    #     error = np.mean((y - prediction_y) ** 2)
    #     ls_error.append(error)
    # print(ls_error)
    # print(np.shape(ls_error))
    # plt.plot(ln_alpha, ls_error, color='blue')
    # plt.show()

    """Homework 4"""
    # C = np.loadtxt('/Users/geminiwenxu/PycharmProjects/ml_plot/Data/C.txt')
    # y = np.loadtxt('/Users/geminiwenxu/PycharmProjects/ml_plot/Data/y.txt')
    # # model = LinearRegression()
    # # model = Ridge(alpha=200)
    # model = linear_model.Lasso(alpha=0.1)
    # model.fit(C, y)
    # s = model.coef_
    # print('shape of s', np.shape(s))
    # reshaped_s = s.reshape((128, 199))
    # print('reshape of s', np.shape(reshaped_s))
    # plt.imshow(reshaped_s, 'BuPu')
    # plt.show()
