import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """Softmax loss function.

    Softmax loss function, naive implementation (with loops)
    Args:
        W: C x D array of weights
        X: D x N array of data. Data are D-dimensional columns
        y: 1-dimensional array of length N with labels 0...K-1, for K classes
        reg: (float) regularization strength
    Returns:
        a tuple of:
        - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    pred = W.dot(X)
    exppred = np.exp(pred)
    n = X.shape[1]
    loss = (reg * np.sum(W ** 2) +
            np.sum(np.log(np.sum(exppred, axis=0)) - pred[y, np.arange(n)]) / n)

    A = exppred / np.sum(exppred, axis=0)
    A[y, np.arange(n)] -= 1

    dW = 2 * reg * W + A.dot(X.T) / n

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    return softmax_loss_naive(W, X, y, reg)
