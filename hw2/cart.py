import numpy as np
import math
from scipy.stats import mode
from multiprocessing.dummy import Pool as ThreadPool


class Forest:
    def __init__(self, num_trees, leaf_size, max_depth, criterion='gini',
                 njobs=4):
        self.num_trees = num_trees
        self.trees = []
        self.leaf_size = leaf_size
        if (max_depth is None):
            self.max_depth = float('inf')
        else:
            self.max_depth = max_depth
        self.criterion = criterion
        self.njobs = njobs

    def _bag(self, X, y):
        sz = len(X)
        idxs = np.random.choice(sz, sz)
        return X[idxs], y[idxs]

    def fit(self, X_train, y_train):
        for i in range(self.num_trees):
            self.trees.append(CART(self.leaf_size, self.max_depth,
                                   self.criterion))

        def f(tree):
            tree.fit(*self._bag(X_train, y_train))
        pool = ThreadPool(self.njobs)
        pool.map(f, self.trees)
        pool.close()
        pool.join()
        return self

    def predict(self, X_test):
        results = []
        for i in range(self.num_trees):
            results.append(self.trees[i].predict(X_test))
        results = np.array(results)
        y_pred = mode(results)[0].reshape(len(X_test),)
        return y_pred


class CART:
    def __init__(self, leaf_size=100, max_depth=10, criterion='gini'):
        self.leaf_size = leaf_size
        if (max_depth is None):
            self.max_depth = float('inf')
        else:
            self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X_train, Y_train):
        self.cart = CARTImpl(X_train, Y_train, self.leaf_size, self.max_depth,
                             self.criterion)

    def predict(self, X_test):
        return self.cart.predict(X_test)

    def dotfile(self):
        return self.cart.dotfile()


class CARTImpl(object):
    def __init__(self, X_train, Y_train, leaf_size=100, max_depth=10,
                 criterion='gini', depth=0, id=0):
        self.id = id
        self.criterion = criterion
        self.leaf_size = leaf_size
        self.X_train = X_train
        self.Y_train = Y_train
        self.num_unique_ys = len(np.unique(Y_train))
        self.left = self.right = None
        self.max_depth = max_depth
        self.depth = depth
        if self._should_terminate():
            self.leaf = True
            self.size = 1
            self.ans = np.argmax(np.bincount(Y_train))
        else:
            self.leaf = False
            self._split()

    def _split(self):
        num_features = self.X_train.shape[1]
        init = True
        best_val = 0
        best_feature = 0
        best_threshold = 0
        for feature in range(num_features):
            elems = np.unique(self.X_train.T[feature])
            for threshold in elems:
                Y_left = self.Y_train[self.X_train.T[feature] <= threshold]
                Y_right = self.Y_train[self.X_train.T[feature] > threshold]
                if (len(Y_left) == 0 or len(Y_right) == 0):
                    continue
                if init:
                    init = False
                    best_val = self._impurity(Y_left, Y_right)
                    best_feature = feature
                    best_threshold = threshold
                else:
                    val = self._impurity(Y_left, Y_right)
                    if (val < best_val):
                        best_val = val
                        best_feature = feature
                        best_threshold = threshold

        self.feature = best_feature
        self.threshold = best_threshold
        X_left = self.X_train[self.X_train.T[best_feature] <= best_threshold]
        X_right = self.X_train[self.X_train.T[best_feature] > best_threshold]
        Y_left = self.Y_train[self.X_train.T[best_feature] <= best_threshold]
        Y_right = self.Y_train[self.X_train.T[best_feature] > best_threshold]
        self.left = CARTImpl(X_left, Y_left, self.leaf_size, self.max_depth,
                             self.criterion, self.depth + 1, self.id + 1)
        id_right = self.id + self.left.size + 1
        self.right = CARTImpl(X_right, Y_right, self.leaf_size, self.max_depth,
                              self.criterion, self.depth + 1, id_right)
        self.size = self.left.size + self.right.size + 1

    def _should_terminate(self):
        return (len(self.Y_train) <= self.leaf_size or
                self.num_unique_ys == 1 or
                self.__gini(self.Y_train) <= 0.001 or
                self.depth >= self.max_depth)

    def __gini(self, y_data):
        total = len(y_data)
        ps = np.bincount(y_data) / float(total)
        ans = 0
        for p in ps:
            ans += p * (1 - p)
        return ans

    def _gini(self, y_left, y_right):
        def f(y_data):
            total = len(y_data)
            ps = np.bincount(y_data) / float(total)
            ans = 0
            for p in ps:
                ans += p * (1 - p)
            return ans
        sz = len(y_left) + len(y_right)
        sz = float(sz)
        return len(y_left)/sz * f(y_left) + len(y_right)/sz * f(y_right)

    def _twoing(self, y_left, y_right):
        n = float(len(y_left) + len(y_right))
        psl = (np.bincount(y_left, minlength=self.num_unique_ys) /
               float(len(y_left)))
        psr = (np.bincount(y_right, minlength=self.num_unique_ys) /
               float(len(y_right)))
        s = np.sum(np.abs(psl - psr)) ** 2
        val = len(y_left) / n * len(y_right) / n * s
        if (val == 0):
            return float('inf')
        else:
            return 1.0 / val

    def _entropy(self, y_left, y_right):
        def f(y_data):
            total = len(y_data)
            ps = np.bincount(y_data) / float(total)
            ans = 0
            for p in ps:
                if (p != 0):
                    ans += p * math.log(p)
            return -ans
        sz = len(y_left) + len(y_right)
        sz = float(sz)
        return len(y_left)/sz * f(y_left) + len(y_right)/sz * f(y_right)

    def _impurity(self, y_left, y_right):
        d = {"gini": self._gini,
             "twoing": self._twoing,
             "entropy": self._entropy}
        try:
            func = d[self.criterion]
        except KeyError:
            raise ValueError("""unknown criterion, possible values:
                                gini, twoing, entropy""")
        return func(y_left, y_right)

    def _predict(self, X_val):
        if (self.leaf):
            return self.ans
        else:
            if (X_val[self.feature] <= self.threshold):
                return self.left._predict(X_val)
            else:
                return self.right._predict(X_val)

    def predict(self, X_test):
        ans = np.zeros(X_test.shape[0])
        for i in range(len(X_test)):
            ans[i] = self._predict(X_test[i])
        return ans

    def dotfile(self):
        prefix = ""
        suffix = ""
        if (self.depth == 0):
            prefix = "strict digraph G{\n"
            suffix = "}"
        if (self.leaf):
            desc = '{id}[label="{ans}"];\n'.format(id=self.id, ans=self.ans)
            return prefix + desc + suffix
        else:
            desc = '{id}[label="x_{var} <= {threshold}"];\n\
{id} -> {left_id}[color=green];\n\
{id} -> {right_id}[color=red];\n'.format(id=self.id, var=self.feature,
                                         threshold=self.threshold,
                                         left_id=self.left.id,
                                         right_id=self.right.id)
            return (prefix +
                    desc + self.left.dotfile() + self.right.dotfile() +
                    suffix)
