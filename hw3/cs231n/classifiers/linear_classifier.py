import numpy as np
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers.softmax import softmax_loss_vectorized


class LinearClassifier:

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        dim, num_train = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.001

        best_loss = -1
        best_W = self.W
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            idx = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[:, idx]
            y_batch = y[idx]
            loss, dW = self.loss(X_batch, y_batch, reg)
            if best_loss == -1 or loss < best_loss:
                best_loss = loss
                best_W = self.W
            self.W -= learning_rate * dW
            loss_history.append(loss)
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f, best_loss: %f' %
                      (it, num_iters, loss, best_loss))

        self.W = best_W
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class. xsq``
        """

        y_pred = np.argmax(self.W.dot(X), axis=0)
        return y_pred

    def predict_prob(self, X):
        y_pred = self.W.dot(X)
        return np.exp(y_pred)/np.sum(np.exp(y_pred), axis=0)

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: D x N array of data; each column is a data point.
        - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
