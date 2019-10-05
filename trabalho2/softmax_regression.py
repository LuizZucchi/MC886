import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse


def get_loss(w, x, y, l):
    m = x.shape[0]
    y_mat = one_hot(y)

    scores = np.dot(x, w)
    prob = softmax(scores)

    loss = (-1/m) * np.sum(y_mat*np.log(prob)) + (l/2) * np.sum(w*w)
    grad = (-1/m) * np.dot(x.T, (y_mat - prob)) + l*w

    return loss, grad


def fix_learn_rate(epoch, init_learn_rate):
    learn_rate = init_learn_rate/(1 + epoch/25)
    return learn_rate


def one_hot(Y):
    m = Y.shape[0]

    one_hot_varient = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    one_hot_varient = np.array(one_hot_varient.todense()).T

    return one_hot_varient


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z.T)/np.sum(np.exp(z), axis=1)).T
    return sm


def get_probs(X, w):
    probs = softmax(np.dot(X, w))

    return probs


def get_predicts(probs):
    predicts = np.argmax(probs, axis=1)
    return predicts


def get_accur(x, y, w):
    probs = get_probs(x, w)
    predict = get_predicts(probs)

    accur = sum(predict == y)/(float(len(y)))

    return accur


def train_softmax(x, y, num_iterations, init_learn_rate):
    w = np.random.randn(x.shape[1], len(np.unique(y)))
    l = 1
    losses = []
    learn_rate = init_learn_rate
    for i in range(0, num_iterations):
        loss, grad = get_loss(w, x, y, l)
        losses.append(loss)
        w = w - (learn_rate * grad)
        learn_rate = fix_learn_rate(i, init_learn_rate)
        print('loss: {}'.format(loss))

    return losses, w


