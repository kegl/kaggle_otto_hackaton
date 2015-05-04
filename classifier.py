import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
 
import os
os.environ["OMP_NUM_THREADS"] = "1"

from lasagne.easy import SimpleNeuralNet
import theano

class BaseClassifier(BaseEstimator):
 
    def __init__(self):
 
        self.clf = Pipeline([
            ('log', LogScaler()),
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[1000],
                                          max_nb_epochs=30,
                                          batch_size=256,
                                          learning_rate=1.,
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
        self.base_classifier.fit(X, y)  
 
    def predict(self, X):
        pred = self.base_classifier.predict(X)
        return self.class_label_encoder_.inverse_transform(pred)  

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = AdaBoostClassifier(
            base_estimator=WeightedBaseClassifier(BaseClassifier()),
            n_estimators=3
        )
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

