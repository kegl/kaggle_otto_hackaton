import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import os
os.environ["OMP_NUM_THREADS"] = "1"

import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class Calibrator(BaseEstimator):
    def __init__(self):
        self.net = None
        self.label_encoder = None
    
    def fit(self, X_array, y_pred_array):
        labels = np.sort(np.unique(y_pred_array))
        num_classes = X_array.shape[1]
        layers0 = [('input', InputLayer),
                   ('dense', DenseLayer),
                   ('output', DenseLayer)]
        X = X_array.astype(theano.config.floatX)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.label_encoder = LabelEncoder()
        #y = class_indicators.astype(np.int32)
        y = self.label_encoder.fit_transform(y_pred_array).astype(np.int32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        num_features = X.shape[1]
        self.net = NeuralNet(layers=layers0,
                             input_shape=(None, num_features),
                             dense_num_units=num_features,
                             output_num_units=num_classes,
                             output_nonlinearity=softmax,

                             update=nesterov_momentum,
                             update_learning_rate=0.01,
                             update_momentum=0.9,

                             eval_size=0.2,
                             verbose=1,
                             max_epochs=100)
        self.net.fit(X, y)

    def predict_proba(self, y_probas_array_uncalibrated):
        num_classes = y_probas_array_uncalibrated.shape[1]
        X = y_probas_array_uncalibrated.astype(theano.config.floatX)
        X = self.scaler.fit_transform(X)
        return self.net.predict_proba(X)