# 2018011423 Ren Yi

import numpy as np
import copy
from tqdm import trange
import warnings
class MyAdaBoostRegressor(object):
    '''
    My implementation of Adaboost with "fit" and "predict" API.
    '''
    def __init__(self, base_estimator, n_estimators = 10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_list = list()
        self.estimator_weight = list()
        
    def fit(self, X, y):
        '''
        train process
        '''
        sample_weight = np.ones(X.shape[0]) * (1 / X.shape[0])
        for t in trange(self.n_estimators):
            now_estimator = copy.deepcopy(self.base_estimator)
            index = np.random.choice(X.shape[0], X.shape[0], replace = True, p = sample_weight)
            now_X, now_Y = X[index, :], np.array(y)[index]
            now_estimator.fit(now_X, now_Y)
            abs_error = np.array([abs(now_estimator.predict(X[i]) - y[i]) for i in range(X.shape[0])])
            abs_error = np.ravel(abs_error)
            max_error = np.max(abs_error)
            sample_error = abs_error / max_error
            error = np.dot(sample_weight, sample_error)
            if error < 0.5:
                self.estimator_list.append(now_estimator)
                beta = error / (1 - error)
                self.estimator_weight.append(np.log(1/beta)) # according to sklearn
                sample_weight = np.array([sample_weight[i] * (beta ** (1.0 - sample_error[i])) for i in range(len(sample_weight))])
                sample_weight /= np.sum(sample_weight)
        return self

    def predict(self, X):
        '''
        give a prediction based on input
        '''
        # Reference to https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/ensemble/_weight_boosting.py#L1015 Line1093
        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimator_list]).T
        # Sort the predictions of each input by the rating score
        sorted_idx = np.argsort(predictions, axis=1)
        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(np.array(self.estimator_weight)[sorted_idx], axis=1, dtype=np.float64)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]
        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]
