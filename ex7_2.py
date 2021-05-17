import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def label(theta):
    return "[%2.2f, %2.2f]" % (theta[0], theta[1])


def generate_data(size):
    x = np.random.uniform(low=-1.0, high=1.0, size=size)
    noise = np.random.normal(loc=0, scale=0.2, size=size)
    t_n_ls = []
    for i in range(size):
        t_n = -0.3 + 0.5 * x[i] + 0.4 * x[i] ** 2 + noise[i]
        t_n_ls.append(t_n)
    return t_n_ls


def model(x, mb):
    for i in range(len(x)):
        return -0.3 + mb[0] * x[i] + mb[1] * x[i] ** 2


def likelihood(t, model, x, w):
    mu = model(x, w)
    ps = stats.norm.pdf(t, mu, 0.2)
    l = 1
    for p in ps:
        l = l * p
    return l


def prior(MB):
    S0 = np.array([[0.5, 0], [0.0, 0.5]])
    m0 = np.array([[0], [0]])
    Prior = stats.multivariate_normal.pdf(MB, m0.ravel(), S0)
    Prior = Prior.reshape(M.shape)
    return Prior


def posterior(prior, likelihood):
    posterior = np.multiply(prior, likelihood)
    return posterior


if __name__ == '__main__':
    # creating 10 data points (x, t)
    N = 10
    x = np.linspace(-1, 1, N)
    t = generate_data(N)

    # draw data
    # plt.plot(x, t, 'k.', markersize=20, label='data points', markeredgecolor='w')
    mb0 = [0.5, 0.4]
    y = -0.3 + 0.5 * x + 0.4 * x ** 2
    # plt.plot(x, y, label='true model')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend()

    # create array to cover parameter space
    res = 100
    M, B = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    MB = np.c_[M.ravel(), B.ravel()]

    # design three in one figure
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    '''row 1'''
    # plot the original prior
    prior_1 = prior(MB)
    # ax2.contourf(M, B, prior_1)
    ax2.set_title('Prior Prob. Dist.')
    ax2.set_xlabel('w_1')
    ax2.set_ylabel('w_2')

    # Plot the 6 sample y(x,w)
    mean = np.zeros(shape=3, dtype=int)
    alpha = 2
    cov = alpha ** (-1) * np.identity(3)
    w = np.random.multivariate_normal(mean, cov, size=100)
    x = np.linspace(-1, 1, 100)
    for i in range(6):
        y = w[i][0] + w[i][1] * x + w[i][2] * x ** 2
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        # plt.plot(x, y)
        plt.title('data space')
        plt.xlabel('x')
        plt.ylabel('y')

    observed_x_1 = [0.7666]
    observed_y_1 = w[1][0] + w[1][1] * x[0] + w[1][2] * x[0] ** 2

    '''row 2'''
    # calculate likelihood function
    L = np.array([likelihood(observed_y_1, model, observed_x_1, mb.reshape(2, 1)) for mb in MB]).reshape(M.shape)

    # draw likelihood function
    # ax1.contourf(M, B, L)
    ax1.set_title('Likelihood')
    ax1.set_xlabel('w_1')
    ax1.set_ylabel('w_2')

    # posterior
    Posterior_1 = posterior(prior_1, L)
    # ax2.contourf(M, B, Posterior_1)
    ax2.set_title('Posterior Prob. Dist.')
    ax2.set_xlabel('w_1')
    ax2.set_ylabel('w_2')

    # Plot the 6 sample y(x,w) with w0 and w1 drawn from the posterior
    w1 = np.linspace(-0.1, 0.7, 10)
    w2 = np.linspace(-0.1, 0.6, 10)
    x = np.linspace(-1, 1, 100)
    for i in range(6):
        y = -0.3 + w1[i] * x + w2[i] * x ** 2
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        # plt.plot(x, y)
        plt.title('data space')
        plt.xlabel('x')
        plt.ylabel('y')

    observed_x_2 = [0.3, 0.5]
    observed_y_2_0 = -0.3 + w1[1] * x[0] + 0.4 * x[0] ** 2
    observed_y_2_1 = -0.3 + w1[1] * x[1] + 0.4 * x[1] ** 2
    observed_y_2 = [observed_y_2_0, observed_y_2_1]

    '''row 3 '''
    # calculate likelihood function
    L = np.array([likelihood(observed_y_2, model, observed_x_2, mb.reshape(2, 1)) for mb in MB]).reshape(M.shape)

    # draw likelihood function
    # ax1.contourf(M, B, L)
    ax1.set_title('Likelihood')
    ax1.set_xlabel('w_1')
    ax1.set_ylabel('w_2')

    # posterior
    Posterior_2 = posterior(Posterior_1, L)
    # ax2.contourf(M, B, Posterior_2)
    ax2.set_title('Posterior Prob. Dist.')
    ax2.set_xlabel('w_1')
    ax2.set_ylabel('w_2')

    # Plot the 6 sample y(x,w) with w0 and w1 drawn from the posterior
    w1 = np.linspace(-0.5, -0.1, 200)
    w2 = np.linspace(0.2, 0.7, 200)
    x = np.linspace(-1, 1, 100)
    # y = -0.3 + w1[1] * x + w2[1] * x ** 2
    # plt.plot(x, y)
    # y = -0.3 + w1[20] * x + w2[20] * x ** 2
    # plt.plot(x, y)
    # y = -0.3 + w1[60] * x + w2[60] * x ** 2
    # plt.plot(x, y)
    # y = -0.3 + w1[100] * x + w2[100] * x ** 2
    # plt.plot(x, y)
    # y = -0.3 + w1[120] * x + w2[120] * x ** 2
    # plt.plot(x, y)
    # y = -0.3 + w1[180] * x + w2[80] * x ** 2
    # plt.plot(x, y)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)

    plt.title('data space')
    plt.xlabel('x')
    plt.ylabel('y')

    observed_x_3 = np.linspace(-1, 1, 200)
    observed_y_3 = []
    for j in range(200):
        temp = -0.3 + w1[j] * observed_x_3[j] + w2[j] * observed_x_3[j] ** 2
        observed_y_3.append(temp)

    '''row 4'''
    # calculate likelihood function
    L = np.array([likelihood(observed_y_3, model, observed_x_3, mb.reshape(2, 1)) for mb in MB]).reshape(M.shape)

    # draw likelihood function
    ax1.contourf(M, B, L)
    ax1.set_title('Likelihood')
    ax1.set_xlabel('w_1')
    ax1.set_ylabel('w_2')

    # posterior
    Posterior_3 = posterior(Posterior_2, L)
    ax2.contourf(M, B, Posterior_3)
    ax2.set_title('Posterior Prob. Dist.')
    ax2.set_xlabel('w_1')
    ax2.set_ylabel('w_2')

    # Plot the 6 sample y(x,w) with w0 and w1 drawn from the posterior
    w0 = np.linspace(-0.31, -0.29, 10)
    w1 = np.linspace(0.51, 0.5, 10)
    x = np.linspace(-1, 1, 100)
    for i in range(6):
        y = -0.3 + w1[i] * x + w2[i] * x ** 2
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.plot(x, y)
        plt.title('data space')
        plt.xlabel('x')
        plt.ylabel('y')

    plt.show()
