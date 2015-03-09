import numpy as np

class KNNLearner:
    #initialize the object with default k = 3
    def __init__(self, k = 3):
        self.k = k
        self.Xtrain = None
        self.Ytrain = None
    #add training data set
    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
    #run the prediction function
    def query(self, Xtest):
        #predicted values array
        Ypredict = np.zeros((Xtest.shape[0], 1), dtype = float)
        #distance array
        distance = np.zeros((self.Xtrain.shape[0], 1), dtype = float)
        #loop through rows in test set
        for i in range(Xtest.shape[0]):
            #calculate euclidean distance between each row in test set and each row in training set
            distance = np.sqrt(np.square(self.Xtrain[:,0] - Xtest[i, 0]) + np.square(self.Xtrain[:,1] - Xtest[i, 1]))
            #select the y values of k nearest neighbors
            knn = [self.Ytrain[ix] for ix in np.argsort(distance)[:self.k]]
            #calculate the average of y values
            Ypredict[i] = np.mean(knn)
        return Ypredict[:,0]