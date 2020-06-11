# 2018011423 Ren Yi

import copy
import numpy as np
from tqdm import trange
class MyBaggingRegressor(object):
    '''
    My implementation of Bagging with "fit" and "predict" API.
    '''
    def __init__(self, base_estimator, n_estimators = 10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_list = list()
    def fit(self, X, y):
        '''
        train process
        '''
        for i in trange(self.n_estimators):
            now_X, now_Y = self.bootstrap(X, y)
            now_estimator = copy.deepcopy(self.base_estimator)
            now_estimator.fit(now_X, now_Y)
            self.estimator_list.append(now_estimator)
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
