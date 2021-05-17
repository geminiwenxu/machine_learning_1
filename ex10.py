import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras import initializers


def generate_data(size):
    mu_1 = [2, 2]
    sigma_1 = [[0.8, 0.4], [0.4, 0.8]]
    mu_2 = [1, -1]
    sigma_2 = [[1.3, -0.7], [-0.7, 1.3]]
    data_1 = np.random.multivariate_normal(mu_1, sigma_1, size)
    data_2 = np.random.multivariate_normal(mu_2, sigma_2, size)
    x = np.concatenate((data_1, data_2))
    return data_1, data_2, x


def label(size):
    t_1 = np.zeros(size)
    t_2 = np.ones(size)
    t_ls = np.concatenate((t_1, t_2))
    return t_ls


def create_model():
    model = Sequential()

    model.add(Dense(2, input_dim=2, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Zeros(), activation=tf.nn.tanh))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01),
                    bias_initializer=initializers.Zeros()))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model


def plot_decision_boundary(X, y, model, epoch_count=0, count=0, steps=1000, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
    # fig.suptitle("Epoch: " + str(epoch_count), fontsize=10)
    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, lw=0)
    fig.savefig("images_new/" + str(count) + "_nn.png")
    return epoch_count


if __name__ == "__main__":
    data_1, data_2, x = generate_data(10)
    y = label(10)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    for i in range(10):
        ax.scatter3D(data_1[i][0], data_1[i][1], y, color="blue")
    for i in range(10):
        ax.scatter3D(data_2[i][0], data_2[i][1], y, color="red")
    plt.title("simple 3D scatter plot")
    plt.show()

    # create model
    model = create_model()

    # train model
    history = model.fit(x, y, epochs=15, batch_size=1)
    weights = model.get_weights()
    print(weights)

    # plot metrics
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    _, _, x_test = generate_data(20)
    y_test = label(20)

    _, accuracy = model.evaluate(x_test, y_test)
    print('Test Accuracy: %.2f' % (accuracy * 100))

    plot_decision_boundary(x_test, y_test, model, cmap='RdBu')
