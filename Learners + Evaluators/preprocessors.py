import numpy as np

def prepend_1s(X):
    M = np.zeros((len(X), (len(X[0])+1)), dtype = float)

    for i in range(len(X)):
        M[i][0] = 1

    for i in range(len(X)):
        for j in range(len(X[0])):
            M[i][j+1] = X[i][j]
    return M

    pass



def poly_lift(X, degree):
    M = np.zeros((len(X),degree), dtype = float)
    for i in range(len(X)):
        for j in range(degree):

            if (j == 0):
                M[i][j] = 1
            if (j != 0 and i != 0):
                M[i][j] = i**j
            if (i == 0 and j != 0):
                M[i][j] = 0
    return M
    pass



def standardize(X):

    minarray = np.zeros(X[0].size, dtype = float)
    maxarray = np.zeros(X[0].size, dtype = float)
    for column in range(len(X[0])):
        max = float("-inf")
        min = float("inf")
        for row in range(len(X)):

            if X[row][column] < min:
                min = X[row][column]
            if X[row][column] > max:
                max = X[row][column]
            if row == (len(X) - 1):
                print("min ", min, "max ", max)
                minarray[column] = min
                maxarray[column] = max
    for column in range(X[0].size):
        for row in range(len(X)):
            if minarray[column] < 0:
                X[row][column] = (X[row][column] + abs(minarray[column]))/(maxarray[column] - minarray[column])

            if minarray[column] == 0:
                    X[row][column] = X[row][column] / maxarray[column]
            if minarray[column] > 0:
                X[row][column] = (X[row][column] - abs(minarray[column])) / (maxarray[column] - minarray[column])

    return X
    pass

