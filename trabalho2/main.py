import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from trabalho2.softmax_regression import train_softmax, get_accur, one_hot
from trabalho2.neural_network import NeuralNetwork
from trabalho2.utils import *

np.set_printoptions(threshold=786432, formatter={'float': lambda x: "{0:0.5f}".format(x)})

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# hiperparametros
data_size = 5000
hidden_size = 100
learn_rate = 0.005
epochs = 300
batch_size = 200
num_batchs = int(data_size/batch_size)

# preparando dados de treino
data_x, data_y = read_data()
data_y = one_hot(data_y)

data_x = np.array(data_x[:data_size])
data_y = np.array(data_y[:data_size])
mean_image = np.mean(np.mean(data_x, axis=1))
data_x = data_x.astype('float32')
data_x = (data_x - mean_image) / np.mean(x, axis=0)

# preparando dados para validação
val_x, val_y = read_data('mini_cinic10/val.npz')

val_x = np.array(val_x[:int(data_size*0.125)])
val_y = np.array(val_y[:int(data_size*0.125)])

mean_image = np.mean(np.mean(val_x, axis=1))
val_x = val_x.astype('float32')
val_x = (val_x - mean_image) / np.mean(x_val, axis=0)

# preparando dados para teste

test_x, test_y = read_data('mini_cinic10/test.npz')
test_x = np.array(test_x[:int(data_size*0.125)])
test_y = np.array(test_y[:int(data_size*0.125)])

mean_image = np.mean(np.mean(test_x, axis=1))
test_x = test_x.astype('float32')
test_x = (test_x - mean_image) / np.mean(x_test, axis=0)


def do_softmax():
    print('Iniciando softmax regression')
    losses, w = train_softmax(data_x, data_y, epochs, learn_rate)
    plt.plot(np.arange(len(losses)), losses)
    plt.show()
    val_accur = get_accur(val_x, val_y, w)
    test_accur = get_accur(test_x, test_y, w)

    print('softmax_regression data_size: {} '
                'hidden_size: {} '
                'learn_rate: {} '
                'batch_size: {} '
                'epochs: {} '
                'val_accur: {} '
                'test_accur: {} '
                'loss: {}\n'.format(data_size,
                                  hidden_size,
                                  learn_rate,
                                  batch_size,
                                  epochs,
                                  val_accur,
                                  test_accur,
                                  losses[-1]
                                  )
                )

    with open('results.txt', 'a') as f:
        f.write('softmax_regression data_size: {} '
                'hidden_size: {} '
                'learn_rate: {} '
                'batch_size: {} '
                'epochs: {} '
                'val_accur: {} '
                'test_accur: {} '
                'loss: {}\n'.format(data_size,
                                  hidden_size,
                                  learn_rate,
                                  batch_size,
                                  epochs,
                                  val_accur,
                                  test_accur,
                                  losses[-1]
                                  )
                )
    with open('softmax_weights.txt', 'w+') as f:
        f.write('coefs: {}'.format(w))
    return


def do_sklearn_nn():

    clf = MLPClassifier(hidden_layer_sizes=(hidden_size,), learning_rate_init=learn_rate, max_iter=epochs, batch_size=batch_size, verbose=True)
    clf.fit(data_x, data_y)

    preds = clf.predict(val_x)
    accur = sum(preds == val_y)/len(val_y)
    print('sklearn val accur: ', accur)

    preds = clf.predict(test_x)

    test_accur = sum(preds == test_y)/len(test_y)
    print('sklearn test accur: ', test_accur)

    losses = clf.loss_curve_

    print('sk_learn data_size: {} '
                'hidden_size: {} '
                'learn_rate: {} '
                'batch_size: {} '
                'epochs: {} '
                'val_accur: {} '
                'test_accur: {} '
                'loss: {}\n'.format(data_size,
                                  hidden_size,
                                  learn_rate,
                                  batch_size,
                                  epochs,
                                  accur,
                                  test_accur,
                                  losses[-1]
                                  )
                )

    with open('results.txt', 'a') as f:
        f.write('sk_learn data_size: {} '
                'hidden_size: {} '
                'learn_rate: {} '
                'batch_size: {} '
                'epochs: {} '
                'val_accur: {} '
                'test_accur: {} '
                'loss: {}\n'.format(data_size,
                                  hidden_size,
                                  learn_rate,
                                  batch_size,
                                  epochs,
                                  accur,
                                  test_accur,
                                  losses[-1]
                                  )
                )
    with open('sklearn_weights.txt', 'w+') as f:
        f.write('coefs: {}'.format(clf.coefs_))

    plt.plot(np.arange(len(losses)), losses)
    plt.savefig('loss_sklearn' + '.png')
    plt.show()


def do_neural_network():
    NN = NeuralNetwork(x[0], y[0], hidden_size, learn_rate, 'relu')

    batch_size = 64

    epochs = 300

    NN.train(x, y, epochs=epochs, num_batchs=num_batchs, batch_size=batch_size, stop_cond=0.001)

    plt.plot(np.arange(len(NN.losses)), NN.losses)
    plt.savefig('loss' + '.png')
    plt.show()

    accur = NN.eval(val_x, val_y)
    print('validation accur: ', accur)

    test_accur, predicts = NN.eval(test_x, test_y)
    print('test accur: ', accur)

    print('NN data_size: {} '
                'hidden_size: {} '
                'learn_rate: {} '
                'batch_size: {} '
                'epochs: {} '
                'val accur: {} '
                'test accur: {} '
                'loss: {}\n'.format(data_size,
                                  hidden_size,
                                  learn_rate,
                                  batch_size,
                                  epochs,
                                  accur,
                                  test_accur,
                                  NN.losses[-1]
                                  )
                )

    with open('results.txt', 'a') as f:
        f.write('data_size: {} '
                'hidden_size: {} '
                'learn_rate: {} '
                'batch_size: {} '
                'epochs: {} '
                'val accur: {} '
                'test accur: {} '
                'loss: {}\n'.format(data_size,
                                  hidden_size,
                                  learn_rate,
                                  batch_size,
                                  epochs,
                                  accur,
                                  test_accur,
                                  NN.losses[-1]
                                  )
                )

    with open('nn_weights.txt', 'w+') as f:
        f.write('W1: {}\nb1: {}\nW2: {}\nb2: {}'.format(NN.W1, NN.b1, NN.W2, NN.b2))

    plot_confusion_matrix(test_y, predicts, classes)


if __name__ == '__main__':
    do_softmax()
    do_sklearn_nn()
    do_neural_network()
