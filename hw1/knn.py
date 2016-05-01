import numpy as np
from scipy import linalg, stats
from sklearn.neighbors import KDTree
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm
import math

def plot_image(img, im_size=28, interpol='none'):
    plt.imshow(img.reshape(im_size, im_size), cmap=matplotlib.cm.Greys_r, interpolation=interpol)

def plot_grid(imgs, nrows, ncols, im_size=28, interpol='none'):
    fig = plt.figure()
    fig.set_size_inches(ncols * 2, nrows * 2)
    for pylab_index, img in enumerate(imgs):
        pylab.subplot(nrows, ncols, pylab_index + 1)
        plt.title(str(pylab_index))
        plot_image(img, im_size, interpol)
        pylab.axis('off')

def accuracy(y_true, y_predict):
    score = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_predict[i]):
            score +=1
    return score / len(y_true)

def cross_validation(X, y, knn, cv_fold=10):
    scores = []
    num_data = X.shape[0]
    ind = np.arange(num_data)
    np.random.shuffle(ind)
    X_part = []
    Y_part = []
    block_size = num_data // cv_fold
    for i in range(cv_fold):
        ind_block = ind[i * block_size : (i + 1) * block_size]
        X_part.append(X[ind_block])
        Y_part.append(y[ind_block])
    X_part = np.array(X_part)
    Y_part = np.array(Y_part)
    for i in range(cv_fold):
        X_train_new = np.concatenate(X_part[np.arange(len(X_part)) != i])
        Y_train_new = np.concatenate(Y_part[np.arange(len(Y_part)) != i])
        X_test_new = X_part[i]
        Y_test_new = Y_part[i]
        knn.fit(X_train_new, Y_train_new)
        Y_pred = knn.predict(X_test_new)
        scores.append(accuracy(Y_pred, Y_test_new))
    return np.mean(scores)

class MatrixBasedKNearestNeighbor(object):
    def __init__(self, num_loops = 0, k = 1, distance = 'l2', kernel='rect'):
        self.dist_mtx = None
        self.num_loops = num_loops
        self.k = k
        self.distance = distance
        self.kernel = kernel

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        num_train_obects = self.X_train.shape[0]
        num_test_obects = X_test.shape[0]
        self.dist_mt = np.zeros((num_test_obects, num_train_obects))

        if self.num_loops == 2:
            if (self.distance == 'l2'):
                for i in range(num_test_obects):
                    for j in range(num_train_obects):
                        self.dist_mt[i][j] = linalg.norm(self.X_train[j] - X_test[i], 2)
            else:
                raise ValueError("Unknown or unsupported metric " + self.distance)

        if self.num_loops == 1:
            if (self.distance == 'l2'):
                for i in range(num_test_obects):
                    self.dist_mt[i] = np.sum((self.X_train - X_test[i]) ** 2, axis=-1) ** 0.5
            elif (self.distance == 'l1'):
                for i in range(num_test_obects):
                    self.dist_mt[i] = np.sum(np.abs(self.X_train - X_test[i]), axis=-1)
            elif (self.distance == 'l3'):
                for i in range(num_test_obects):
                    self.dist_mt[i] = np.sum(np.abs(self.X_train - X_test[i]) ** 3, axis=-1) ** (1.0 / 3)
            else:
                raise ValueError("Unknown or unsupported metric " + self.distance)


        if self.num_loops == 0:
            if (self.distance == 'l2'):
                A = np.sum(self.X_train ** 2, axis=1)
                B = np.sum(X_test ** 2, axis=1)
                self.dist_mt = ((A - 2 * X_test.dot(self.X_train.T)).T + B).T
            elif (self.distance == 'cosine'):
                A = np.sum(self.X_train ** 2, axis=1) ** 0.5
                B = np.sum(X_test ** 2, axis=1) ** 0.5
                C = self.X_train.dot(X_test.T)
                self.dist_mt = 1 - (C / B).T / A.T
            else:
                raise ValueError("Unknown metric or unsupported metric" + self.distance)

        return self.predict_labels()

    def get_kernel(self):
        if (self.kernel == 'rect'):
            return lambda x: 0.5 if (abs(x) <= 1) else 0
        if (self.kernel == 'opt'):
            return lambda r: 3.0 / 4 * (1 - r ** 2) if abs(r) <= 1 else 0
        if (self.kernel == 'triang'):
            return lambda r: 1 - abs(r) if abs(r) <= 1 else 0
        if (self.kernel == 'gauss'):
            return lambda r: (math.pi * 2) ** (-0.5) * math.exp( -0.5 * (r ** 2))
        if (self.kernel == 'quart'):
            return lambda r: 15.0 / 16 * (1 - r ** 2) ** 2 if (abs(r)) <= 1 else 0
        return None

    def predict_labels(self):
        num_test = self.dist_mt.shape[0]
        y_pred = np.zeros(num_test)
        K = self.get_kernel()
        for i in range(num_test):
            idxs = np.argsort(self.dist_mt[i])[:self.k]
            closest_y = self.y_train[idxs]
            dists = self.dist_mt[i][idxs]
            max_dist = 1 + dists[-1]
            best_ans = -1
            best_value = 0

            for digit in range(10):
                value = 0
                for j in range(self.k):
                    if (closest_y[j] == digit):
                        value += K(dists[j] / max_dist)
                if (value >= best_value):
                    best_value = value
                    best_ans = digit
            y_pred[i] = best_ans

        return y_pred

class KDBasedKNearestNeighbor(object):
    def __init__(self, k = 1):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.kd_tree = KDTree(X_train)

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        num_test = X_test.shape[0]
        for i in range(num_test):
            _, idxs = self.kd_tree.query(X_test[i].reshape(1, -1), k=self.k)
            closest_y = self.y_train[idxs[0]]
            a, _ = stats.mode(closest_y)
            y_pred[i] = a[0]
        return y_pred
