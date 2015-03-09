import numpy as np

class LinRegLearner():
    def __init__(self):
        self.Xtrain = None
        self.Ytrain = None
        self.m = None
        self.c = None

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        #append a column of 1s to your Xtrain matrix
        A = np.hstack([self.Xtrain, np.ones((len(self.Xtrain[:, 0]), 1))])
        self.m = np.linalg.lstsq(A, self.Ytrain)[0][:-1]
        self.c = np.linalg.lstsq(A, self.Ytrain)[0][-1]
        
    def query(self, Xtest):
        return np.dot(Xtest, self.m) + self.c