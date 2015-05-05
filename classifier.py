import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
 
import os
os.environ["OMP_NUM_THREADS"] = "1"
 
import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.updates import nesterov_momentum
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
 
 
class BaseClassifier(BaseEstimator):
 
    def __init__(self):
        self.net = None
        self.label_encoder = None

    def fit(self, X, y):
        layers0 = [('input', InputLayer),
                   ('dense0', DenseLayer),
                   ('dropout', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('output', DenseLayer)]
        X = X.astype(theano.config.floatX)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y).astype(np.int32)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        num_classes = len(self.label_encoder.classes_)
        num_features = X.shape[1]
        self.net = NeuralNet(layers=layers0,
                             input_shape=(None, num_features),
                             dense0_num_units=300,
                             dense0_nonlinearity=rectify,
                             dropout_p=0.5,
                             dense1_num_units=200,
                             dense1_nonlinearity=rectify,
                             output_num_units=num_classes,
                             output_nonlinearity=softmax,
 
                             update=adagrad,
                             update_learning_rate=0.02,
 
                             eval_size=0.2,
                             verbose=1,
                             max_epochs=20,
                             )
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = X.astype(theano.config.floatX)
        X = self.scaler.fit_transform(X)
        return self.label_encoder.inverse_transform(self.net.predict(X))
 
    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        X = self.scaler.fit_transform(X)
        return self.net.predict_proba(X)
 
class WeightedBaseClassifier(BaseEstimator):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
 
    def fit(self, X, y, sample_weight, n_sample = 0):
        self.class_label_encoder_ = LabelEncoder()
        self.class_label_encoder_.fit(y)
        self.classes_ = self.class_label_encoder_.classes_
        if n_sample == 0:
            n_sample = X.shape[0]
        sample_indexes = np.random.choice(
            range(X.shape[0]), size=n_sample, p=sample_weight)
        X_sampled = X[sample_indexes]          
        y_sampled = y[sample_indexes]
        sampled_label_encoder = LabelEncoder()
        sampled_label_encoder.fit(y_sampled)
        self.sampled_class_indexes_ = np.array(
            [np.nonzero(self.class_label_encoder_.classes_ == sampled_class)[0][0]
             for sampled_class in sampled_label_encoder.classes_])
        self.base_classifier.fit(X_sampled, y_sampled)  
 
    def predict(self, X):
        pred = self.base_classifier.predict(X)
        return self.class_label_encoder_.inverse_transform(pred)  
 
    def predict_proba(self, X):
        proba = np.zeros([X.shape[0], self.class_label_encoder_.classes_.shape[0]])
        proba[:,self.sampled_class_indexes_] = self.base_classifier.predict_proba(X)
        return proba
 
class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = AdaBoostClassifier(
            base_estimator=WeightedBaseClassifier(BaseClassifier()),
            n_estimators=10
        )
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
