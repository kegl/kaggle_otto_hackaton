import numpy as np

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        #print X_dict.keys()
        cols = X_dict[0].keys()[1:] # this takes everything except id
        return np.array([[instance[col] for col in cols] for instance in X_dict])