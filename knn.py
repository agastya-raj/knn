import numpy as np
import scipy
from scipy import stats
from scipy.spatial.distance import cdist

class KNN(object):
    """KNN Classifier with L2 distance"""
    def __init__(self):
        pass
    
    """X: a numpy array of shape (examples, dimensions)
     Y: a numpy array of class labels of shape (examples, 1)"""
    def train(self, X, Y):
        self.xtrain = X
        self.ytrain = Y
        
    """X is the test data, k is the number of nearest neighbors"""
    def predict(self, X, vectorized=True, k=3):
        if vectorized == True:
            dists = self.vectorized_distance(X)
        elif vectorized == False:
            dists = self.explicit_distance(X)
        return self.predict_labels(dists, k=k)
        
    """The L2 distance between the point and the neighbors is calculated."""
    def explicit_distance(self, X):
        test_size = X.shape[0]
        train_size = self.xtrain.shape[0]
        dists = np.zeros((test_size, train_size))
        for i in range(test_size):
            dists[i,:] = np.sqrt(np.sum(np.square(self.xtrain - X[i,:])), axis=1)
        return dists
    
    """Vectorized implementation of L2 distance calculation"""
    def vectorized_distance(self, X):
        test_size = X.shape[0]
        train_size = self.xtrain.shape[0]
        dists = np.transpose(cdist(self.xtrain, X, 'euclidean'))
        return dists
    
    def predict_labels(self, dists, k=1):
        predictions = np.zeros((dists.shape[0],))
        for i in range(dists.shape[0]):
            labels = np.argsort(dists[i,:])[:k]
            predictions[i] = int(stats.mode(self.ytrain[labels])[0][0])
        return predictions

