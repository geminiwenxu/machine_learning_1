import matplotlib.pyplot as plt
import numpy as np


def generate_data():
    mu_1 = [2, 2]
    sigma_1 = [[0.5, 0.9], [0.9, 0.3]]
    mu_2 = [3, 3]
    sigma_2 = [[0.5, -0.6], [-0.6, 0.5]]
    data_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
    data_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)
    x = np.concatenate((data_1, data_2))
    x0 = np.ones(200).reshape(200, 1)
    x = np.append(x0, x, axis=1)
    return data_1, data_2, x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def phi(data_point):
    return [data_point[0], data_point[1], data_point[2]]


def w_phi(data_point, w):
    return w[0] * data_point[0] + w[1] * data_point[1] + w[2] * data_point[2]


def hypothesis(data_point, w):
    y = sigmoid(w_phi(data_point, w))
    print(y)
    return y


def label():
    # t_ls = []
    # for i in range(200):
    #     y = hypothesis(x[i], [-1, 0.2, 0.3])
    #     print(y)
    #     if y >= 0.5:
    #         t = 1
    #     else:
    #         t = 0
    #     t_ls.append(t)
    # return t_ls

    t_1 = np.zeros(100)
    t_2 = np.ones(100)
    t_ls = np.concatenate((t_1, t_2))
    return t_ls


def error(x, w, t_ls):
    E = 0
    for i in range(200):
        E += np.dot((hypothesis(x[i], w) - t_ls[i]), phi(x[i]))
    return E


if __name__ == '__main__':
    '''generating data and label'''
    data_1, data_2, x = generate_data()
    t_ls = label()

    '''plotting the true data points and the classifier'''
    # for i in range(200):
    #     y = hypothesis(x[i], [-1, 0.2, 0.3])
    #     if y >= 0.5:
    #         t = 1
    #         plt.scatter(x[i][1], x[i][2], color='r')
    #     else:
    #         t = 0
    #         plt.scatter(x[i][1], x[i][2], color='b')
    for i in range(100):
        plt.scatter(data_1[i][0], data_1[i][1], color='green')
        plt.scatter(data_2[i][0], data_2[i][1], color='red')

    '''gradient descent method'''
    w = [-0.1, 0.1, 0.2]
    alpha = 0.01
    # iterations = 5
    # for _ in range(iterations):
    #     error_value = error(x, w, t_ls)
    #     w = w - np.dot(error_value, alpha)
    # x_2 = np.linspace(-2, 5, 100)
    # x_1 = -w[0] / w[1] - (w[2] / w[1]) * x_2
    # plt.plot(x_1, x_2, color='pink', label='iter=5')
    #
    # iterations = 50
    # for _ in range(iterations):
    #     error_value = error(x, w, t_ls)
    #     w = w - np.dot(error_value, alpha)
    # x_2 = np.linspace(-2, 5, 100)
    # x_1 = -w[0] / w[1] - (w[2] / w[1]) * x_2
    # plt.plot(x_1, x_2, color='red', label='iter=50')

    iterations = 500
    for _ in range(iterations):
        error_value = error(x, w, t_ls)  # error value of running through all 200 data points
        w = w - np.dot(error_value, alpha)
    x_2 = np.linspace(-5, 5, 100)
    x_1 = -w[0] / w[1] - (w[2] / w[1]) * x_2
    print(w)
    plt.plot(x_1, x_2, color='r', label='iter=5000')
    # plt.title(f"classification of iterations= {iterations}")
    # plt.title("plotting classifier with alpha =0.05")
    plt.show()
