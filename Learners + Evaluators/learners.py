import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class Regressor(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class NotYetTrainedException(Exception):
    """Learner must be trained (fit) before it can predict points."""
    pass


def simple_kernel(x1, x2):
    return (np.dot(x1, x2) + 1) ** 2


class ToyRegressor(Regressor):
    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = np.average(y)

    def predict(self, X):
        if self.mean is not None:
            return np.array([self.mean for _ in X])
        else:
            raise NotYetTrainedException
        pass


class OLS(Regressor):

    def __init__(self):
        self.theta = np.array

        pass

    def fit(self, X, y):
        firstterm = np.dot(np.transpose(X), X)
        secondterm = np.dot(np.transpose(X), y)
        self.theta = np.linalg.solve(firstterm, secondterm)

        pass

    def predict(self, X):
        if self.theta is not None:
            return self.theta.dot(X)
        else:
            raise NotYetTrainedException
        pass


class RidgeRegression(Regressor):

    def __init__(self, lamb):

        self.regterm = lamb

        self.modelparameters = None
        pass

    def fit(self, X, y):

        matrix = [1 / len(X) * l for l in np.transpose(X).dot(X)]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (i == j):
                    matrix[i][i] = matrix[i][i] + self.regterm
        secondterm = [1 / len(X) * j for j in np.transpose(X).dot(y)]
        self.modelparameters = np.linalg.solve(matrix, secondterm)

        pass

    def predict(self, X):
        if self.modelparameters is not None:
            return np.dot(self.modelparameters,X)
        else:
            raise NotYetTrainedException

        pass


class GeneralizedRidgeRegression(Regressor):
    def __init__(self, regweights):

        self.regterm = regweights

        self.modelparameters = None
        pass

    def fit(self, X, y):

        matrix = [1 / len(X) * l for l in np.transpose(X).dot(X)]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (i == j):
                    matrix[i][i] = matrix[i][i] + self.regterm[i]
        secondterm = [1 / len(X) * j for j in np.transpose(X).dot(y)]
        self.modelparameters = np.linalg.solve(matrix, secondterm)

        pass

    def predict(self, X):
        if self.modelparameters is not None:
            return np.dot(self.modelparameters, X)
        else:
            raise NotYetTrainedException

        pass

class DualRidgeRegression(Regressor):
    def __init__(self, lamb, kernel):

        self.regterm = lamb
        self.kern = kernel
        self.modelparameters = None

        pass

    def fit(self, X, y):
        K = np.zeros((len(X), len(X)), dtype = float)

        for rows in range(len(X)):
            for columns in range(len(X)):
                K[rows][columns] = self.kern(X[rows], X[columns]) #big K
        for i in range(len(X)):
            K[i][i] = K[i][i] + self.regterm #big K + lambda value
        self.modelparameters = np.linalg.solve(K, y) #a value

        pass

    def smallk(self, xvalue, X):
        tobereturned = np.zeros((len(X), len(self.kern(X[0], xvalue))), dtype = float)
        for i in range(len(X)):
            tobereturned[i] = self.kern(X[i], xvalue)
        return tobereturned
        
    def predict(self, X):
        tobereturned = np.zeros((len(X), 1), dtype = float)

        for j in range(len(X)):
            tobereturned[j] = self.modelparameters.transpose().dot(self.smallk(j, X))
        if self.modelparameters is not None:
            return tobereturned
        else:
            raise NotYetTrainedException
        pass


class AdaptiveLinearRegression(Regressor):
    def __init__(self, kernel):  # note kernel used in totally different way
        self.kern = kernel
        self.theta = None
        self.ymember = np.array([])
        self.smallx = None
        pass

    def removed(self, toberemoved, index):
        return np.delete(toberemoved, index, axis = 0)
        pass

        pass
    
    def isolatedx(self, X, index):
        containerarray = np.array([])
        for i in range(len(X)):
            if i!= index:
                containerarray.append(i)
        return np.delete(X, containerarray, axis = 0)

    def fit(self, X, y):
        K = np.zeros((len(X), len(X)), dtype = float)
        if self.smallx == None:
            self.ymember = y
        else:
            for rows in range(len(X)):
                for columns in range(len(X)):
                    K[rows][columns](self.kern(X[rows], X[columns]))  # big K
                    print(self.kern(self.smallx, X[rows]))
            self.theta = np.linalg.solve(X.transpose().dot(K).dot(X), X.transpose().dot(K).dot(y))
            self.ymember = y
        pass

    def predict(self, X):
        toReturn = np.zeros((len(X),1), dtype = float)
        thetamatrix = np.zeros((len(X), X[0].size), dtype = float)  #nxd theta matrix
        rowtheta = np.zeros((1, X[0].size), dtype = float)
        for rows in range(len(X)):
            self.smallx = self.isolatedx(X, rows)
            fit(self.removed(X, rows), self.removed(self.ymember, rows))
            thetamatrix[rows] = self.theta.transpose()

        for i in range(len(X)):
            toReturn[i] = thetamatrix[i].transpose().dot(X[i])

        return toReturn
        pass









