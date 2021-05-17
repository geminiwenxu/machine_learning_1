import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math


def clt(k):
    batch_size = k
    print("the batch size is: ", batch_size)
    means = []
    for i in range(0, 1000):
        x = np.random.uniform(low=0.0, high=1.0, size=batch_size)
        mean = sum(x) / batch_size
        means.append(mean)
    means = means
    plt.figure(figsize=(20, 10))
    plt.hist(means, bins=20, range=(0, 1), color='purple', density=True, stacked=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'k={k}')

    plt.xticks(np.arange(0, 1.05, step=0.05))
    plt.show()


def posterior(n):
    size = n
    x = np.random.normal(loc=2, scale=2, size=n)
    x_mean = sum(x) / n

    # prior parameters
    mu_0 = 0
    sigma_0 = math.sqrt(8)
    sigma = 2

    mean = (n * ((sigma_0) ** 2) * x_mean) / (n * (sigma_0 ** 2) + sigma ** 2) + (sigma ** 2 * mu_0) / (
            n * sigma_0 ** 2 + sigma ** 2)
    variance = 1 / ((1 / sigma_0 ** 2) + n / sigma ** 2)

    # to compute a gaussian distribution:
    x_min = -10.0
    x_max = 10.0

    mean = mean
    std = math.sqrt(variance)

    x = np.linspace(x_min, x_max, 100)

    y = scipy.stats.norm.pdf(x, mean, std)

    plt.plot(x, y, color='coral')

    plt.grid()

    plt.xlim(x_min, x_max)
    plt.ylim(0, 2)


def prior():
    x_min = -10.0
    x_max = 10.0

    mean = 0
    std = math.sqrt(8)

    x = np.linspace(x_min, x_max, 100)

    y = scipy.stats.norm.pdf(x, mean, std)

    plt.plot(x, y, color='blue')

    plt.grid()

    plt.xlim(x_min, x_max)
    plt.ylim(0, 2)


def final():
    prior()
    posterior(1)
    posterior(10)
    posterior(50)
    posterior(100)
    plt.show()


if __name__ == '__main__':
    clt(k=1)
    final()
    print()

