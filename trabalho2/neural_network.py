"""
O NUMERO DE COLUNAS NA MATRIX A TEM QUE SER IGUAL AO DE LINHAS NA MATRIX B A_COLS == B_ROWS
"""
from MC886.trabalho2.utils import *


class NeuralNetwork:
    def __init__(self, x, y, size_hidden, learning_rate, func):
        self.x = x
        self.y = y
        self.output = np.zeros((y.shape[0], 1))
        self.W1 = np.random.randn(x.shape[0], size_hidden)
        self.W2 = np.random.randn(size_hidden, y.shape[0])
        self.l1 = np.zeros((size_hidden, 1))
        self.b1 = np.ones((1, size_hidden))
        self.b2 = np.ones((1, y.shape[0]))
        self.init_learn_rate = learning_rate
        self.learn_rate = learning_rate
        self.losses = []
        self.func = func
        return

    def feedforward(self):
        if self.func == 'sigmoid':
            self.l1 = sigmoid(np.dot(self.x, self.W1) + self.b1)
            self.output = softmax(np.dot(self.l1, self.W2) + self.b2)
        if self.func == 'relu':
            self.l1 = relu(np.dot(self.x, self.W1) + self.b1)
            self.output = softmax(np.dot(self.l1, self.W2) + self.b2)
        else:
            print('invalid func, use sigmoid or relu')
            return
        return self.output

    def fix_learn_rate(self, epoch):
        self.learn_rate = self.init_learn_rate/(1 + epoch/25)
        return

    def backprop(self):
        """
        d_loss_: derivada do custo em relação a um parametro qualquer (e.g w1, w2, b1, etc)
        :return:
        """
        diff = self.output - self.y

        d_loss_w2 = np.dot(self.l1.T, diff)
        d_loss_b2 = diff

        z1 = np.dot(self.x, self.W1) + self.b1
        d_loss_l1 = np.dot(diff, self.W2.T)

        if self.func == 'sigmoid':
            d_z1 = d_sigmoid(z1)
            d_loss_w1 = np.dot(np.array([self.x]).T, d_z1 * d_loss_l1)
            d_loss_b1 = d_loss_l1 * d_z1
        if self.func == 'relu':
            d_z1 = d_relu(z1)
            d_loss_w1 = np.dot(np.array([self.x]).T, d_z1 * d_loss_l1)
            d_loss_b1 = d_loss_l1 * d_z1
        else:
            print('invalid func, use sigmoid or relu')
            return
        # print('\n ANTES \n')
        # print('W1: {}'.format(self.W1))
        # print('W2: {}'.format(self.W2))
        # print('b1: {}'.format(self.b1))
        # print('b2: {}'.format(self.b2))
        self.W1 -= d_loss_w1 * self.learn_rate
        self.b1 -= d_loss_b1.sum(axis=0) * self.learn_rate
        self.W2 -= d_loss_w2 * self.learn_rate
        self.b2 -= d_loss_b2.sum(axis=0) * self.learn_rate
        # print('\n DEPOIS \n')
        # print('W1: {}'.format(self.W1))
        # print('W2: {}'.format(self.W2))
        # print('b1: {}'.format(self.b1))
        # print('b2: {}'.format(self.b2))
        return

    def momentum(self, X, Y, gamma, batch_size):
        return

    def train(self, X, Y, epochs, num_batchs, batch_size, stop_cond):
        for epoch in range(epochs):
            p = np.random.permutation(X.shape[0])
            X = X[p]
            Y = Y[p]
            losses_aux = []
            for i in range(num_batchs):
                outputs = self.feedforward()
                x_batch = X[batch_size * i:batch_size * i + batch_size]
                y_batch = Y[batch_size * i:batch_size * i + batch_size]
                for j in range(x_batch.shape[0]):
                    self.x = x_batch[j]
                    self.y = y_batch[j]
                    if j != 0:
                        outputs = np.append(outputs, self.feedforward(), axis=0)
                    self.backprop()
                losses_aux.append(cross_entropy(y_batch, outputs))
            loss = np.mean(losses_aux)
            self.losses.append(loss)
            self.fix_learn_rate(epoch)
            if epoch % 5 == 0:
                eval, a = self.eval(X[:int(X.shape[0]*0.1)], Y[:int(Y.shape[0]*0.1)])
                print('epoch: {}, Loss: {}, accur: {}'.format(epoch, loss, eval))
            if epoch > 10:
                if np.all(np.abs(np.diff(self.losses[epoch-5:-1])) < stop_cond):
                    print('Loss did not improve in last 5 epochs, stopping')
                    return

    def eval(self, x, y):
            right = 0
            for i in range(y.shape[0]):
                self.x = x[i]
                predict = self.feedforward()
                if i == 0:
                    predicts = predict
                else:
                    predicts = np.append(predicts, predict, axis=0)
                aux = np.zeros_like(predict)
                aux[np.arange(len(predict)), predict.argmax(1)] = 1
                a = y[i]
                b = aux[0]
                if np.array_equal(aux[0], y[i]):
                    right += 1
                    # print('\n{} predict: {}'.format(right, aux[0]))
                    # print('{} actual: {}\n'.format(right, y[i]))
            return right/y.shape[0], predicts


