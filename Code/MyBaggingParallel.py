# 2018011423 Ren Yi

import copy
import numpy as np
import concurrent.futures
from tqdm import trange
class MyBaggingParallelRegressor(object):
    '''
    My implementation of Parallel Bagging with "fit" and "predict" API.
    '''
    def __init__(self, base_estimator, n_estimators = 10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_list = list()
        self.X = np.array(0)
        self.y = np.array(0)
        
    def _single_fit(self, idx):
        now_estimator = copy.deepcopy(self.base_estimator)
        now_estimator.fit(self.X[idx], self.y[idx])
        return now_estimator

    def _get_bootstrapped_idx(self, length):
        return [np.random.choice(length, length, replace = True).astype(int) for i in range(self.n_estimators)]

    def fit(self, X, y):
        '''
        train process
        '''
        self.X = X
        self.y = np.array(y)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self.estimator_list = list(executor.map(self._single_fit, self._get_bootstrapped_idx(X.shape[0])))
        return self

    def predict(self, X):
        '''
        give a prediction based on input
        '''
        label = np.zeros(X.shape[0])
        for estimator in self.estimator_list:
            label += estimator.predict(X)
        label /= self.n_estimators
        return label
    def bootstrap(self, X, y):
        '''
        give bootstrapped samples
        '''
        index = np.random.choice(X.shape[0], X.shape[0], replace = True).astype(int)
        return X[index, :], np.array(y)[index]
