import numpy as np


# A decorator -- don't worry if you don't understand this.
# It just makes it so that each loss function you implement automatically checks that arguments have the same number of elements


def loss_fun(fun):
    def toR(y_true, y_preds):
        n, = y_true.shape
        npreds, = y_preds.shape
        assert n == npreds, "There must be as many predictions as there are true values"
        return fun(y_true, y_preds)

    return toR


@loss_fun
def zero_one(y_true, y_preds):
    n, = y_true.shape
    return np.sum([1 for yt, yp in zip(y_true, y_preds) if yt == yp]) / n


@loss_fun
def MSE(y_true, y_preds):
    squared_sum = 0

    for i in range(len(y_true)):
        squared_sum += (y_true[i] - y_preds[i]) ** 2
    return squared_sum / len(y_true)

    pass


@loss_fun
def MAD(y_true, y_preds):
    abs_sum = 0
    for i in range(len(y_true)):
        abs_sum += abs(y_true[i] - y_preds[i])
    return abs_sum / len(y_true)

    pass


def removeindex(vector, numfolds, bool, foldtodelete):
    containerarray = []
    if (bool == 1):  # returning points not in the test fold
        for i in range(len(vector)):
            if (i % numfolds == foldtodelete):
                containerarray.append(i)


    if (bool == 0):  # returning points in the test fold
        for j in range(len(vector)):
            if (j % numfolds != foldtodelete):
                containerarray.append(j)  # delete all points which aren't in fold

    toreturn = np.delete(vector, containerarray, axis = 0)
    return toreturn

def cross_validation(X, y, reg, evaler, num_folds=10):
    counter = 0.0

    for j in range(num_folds):  # traverse through all the folds
        remainingx = removeindex(X, num_folds, 1, j)  # x values not in the test fold
        remainingy = removeindex(y, num_folds, 1, j)  # y values not in the test fold
        testx = removeindex(X, num_folds, 0, j)  # x values in the test fold
        testy = removeindex(y, num_folds, 0, j)  # y values in the test fold
        reg.fit(remainingx, remainingy)  # learn a model based on training data point
        predictedys = reg.predict(testx)
        counter += evaler(testy, predictedys)  # total loss
    return counter/num_folds
    pass



