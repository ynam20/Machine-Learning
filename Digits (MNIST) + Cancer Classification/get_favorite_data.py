import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss


def get_favorite_data():
    # your code here
    # Remember, this should return a single point (x, y)
    # and you are not allowed to store state -- every call to this
    # function should return an iid sample from the P you're defining.

    d = 8

    mu0 = np.array([-2 for i in range(d)])
    x = np.random.multivariate_normal(mean=mu0, cov=np.eye(d))
    if np.linalg.norm(x - mu0) > 3.5:
        y = 1
    else:
        if np.linalg.norm(x - mu0) > 2.5:
            y = 0
        else:
            if np.linalg.norm(x-mu0) > 1.5:
                y = 2
            else:
                if np.linalg.norm(x-mu0) > 1:
                    y = 3
                else:
                    y = 4

    return (x, y)

def example_get_favorite_data():
    # Two, far apart, spherical Gaussian blobs
    d = 5

    mu0 = np.array([-5 for i in range(d)])
    mu1 = np.array([5 for i in range(d)])

    y = np.random.binomial(1, 0.5)  # flip a coin for y

    if y == 0:
        x = np.random.multivariate_normal(mean=mu0, cov=np.eye(d))
    else:
        x = np.random.multivariate_normal(mean=mu1, cov=np.eye(d))

    return x, y

def get_lots_of_favorite_data(n, data_fun = get_favorite_data()):
    pts = [data_fun() for _ in range(n)]
    Xs, ys = zip(*pts)
    X = np.array(Xs)
    y = np.array(ys)
    return (X, y)


if __name__ == "__main__":

    print("And here we use get_lots_of_favorite_data to obtain X and y:")
    X, y = get_lots_of_favorite_data(200, get_favorite_data)

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.25, random_state=42)

    for i in range(5):
        clf = KNeighborsClassifier(n_neighbors=i + 1)  # some classifier that you fit on some data
        clf.fit(X_trn, y_trn)
        ytestpredicted = clf.predict(X_tst)
        print(zero_one_loss(y_tst, ytestpredicted))

    for i in range(5):
        if i == 4:
            clf1 = DecisionTreeClassifier("entropy")  # some classifier that you fit on some data
        else:
            clf1 = DecisionTreeClassifier("entropy", max_depth=i + 1)
        clf1.fit(X_trn, y_trn)
        ytestpredicted = clf1.predict(X_tst)
        print(zero_one_loss(y_tst, ytestpredicted))
    for i in range(3):

        if i == 0:
            clf2 = SVC(kernel='linear')  # some classifier that you fit on some data

        else:
            if i == 1:
                clf2 = SVC(kernel='rbf')
                print("LAURA ")

            else:
                clf2 = SVC(kernel='poly')
                print("ENEMY")
        clf2.fit(X_trn, y_trn)
        ytestpredicted = clf2.predict(X_tst)

        print(zero_one_loss(y_tst, ytestpredicted))


    print("X:")
    print(X)
    print("y:")
    print(y)