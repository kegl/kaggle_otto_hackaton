from sklearn.base import BaseEstimator

import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import theano


class Classifier(BaseEstimator):

    def __init__(self):

        self.clf = Pipeline([
#            ('log', LogScaler()),
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[50],
                                          max_nb_epochs=30,
                                          batch_size=128,
                                          learning_rate=0.1,
                                          momentum=0.3,
                                          validation_set_ratio=0.1,
                                          patience_threshold_progression_rate=0.001,
                                          patience_stat="valid_loss",
                                          optimization_method='rmsprop',
                                          L1_factor=0.0001)),
        ])

    def fit(self, X, y):
        X = X.astype(theano.config.floatX)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict_proba(X)

class LogScaler(object):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(1 + X)
