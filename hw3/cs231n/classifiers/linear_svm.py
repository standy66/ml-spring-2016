import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...C-1, for C classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    classes = W.shape[0]
    n = X.shape[1]
    dW = np.zeros(W.shape)
    loss = 0

    predict = W.dot(X)

    for i in range(n):
        for c in range(classes):
            if c == y[i]:
                continue
            margin = 1 + predict[c][i] - predict[y[i]][i]
            if margin > 0:
                loss += margin
                dW[c] += X.T[i]
                dW[y[i]] -= X.T[i]
    dW /= n
    loss /= n

    dW += 2 * reg * W
    loss += reg * np.sum(W ** 2)
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = 2 * reg * W
    n = X.shape[1]
    predict = W.dot(X)
    margin = 1 + predict - predict[y, np.arange(n)]
    margin[margin <= 0] = 0
    loss = np.sum(margin) / n + reg * np.sum(W ** 2) - 1
    margin[margin > 0] = 1
    margin[y, np.arange(n)] = 1 - np.sum(margin, axis=0)
    dW += margin.dot(X.T) / n

    return loss, dW
