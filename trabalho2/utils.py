import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from scipy import sparse


def get_one_hot(y):
    m = y.shape[0]

    one_hot = sparse.csr_matrix((np.ones(m), (y, np.array(range(m)))))
    one_hot = np.array(one_hot.todense()).T

    return one_hot


def softmax(z):
    exp_z = np.exp(z - np.max(z))  # normalize to avoid overflow

    return exp_z/exp_z.sum(axis=1, keepdims=True)


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    sig = sigmoid(z)

    return sig[0] * (1 - sig[0])


def relu(z):
    z = np.clip(z, -500, 500)
    return z * (z > 0)


def d_relu(z):
    z = np.clip(z, -500, 500)
    return 1. * (z > 0)


def cross_entropy(actual, result):
    n = actual.shape[0]

    return np.sum(-actual * np.log(result + 1e-9))/n


def plot_confusion_matrix(actual, result, classes, title='Confusion Matrix', normalize=False, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
    cm = confusion_matrix(actual, result)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(actual, result)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def rgb_2_gray(x):
    r = x[0: 1024] * 0.2125
    g = x[1024: 2048] * 0.7154
    b = x[2048: 3072] * 0.0721
    a = np.add(g, b)
    gray_img = np.add(r, a)/255

    return gray_img

def read_data(path='mini_cinic10/train.npz'):
    data = np.load(path)

    return data['xs'], data['ys']

