from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.random_projection import SparseRandomProjection

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('rp', SparseRandomProjection(n_components=100, density=0.5)),
            ('rf', AdaBoostClassifier(
                base_estimator=RandomForestClassifier(max_depth=5, n_estimators=20),
                n_estimators=10)
            )
        ])
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)