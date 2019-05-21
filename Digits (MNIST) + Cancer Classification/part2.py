import numpy as np
import dill as pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.svm import SVC
# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=DeprecationWarning)




if __name__ == "__main__":
    assert 1/2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 0.20.0)")


import matplotlib.pyplot as plt
def plot_decision_boundary(ax, clf, x_min = 0, x_max = 1, y_min = 0, y_max = 1, res=0.01, cm = plt.cm.RdBu):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.5, cmap=cm)

def plot_num(ax, x):
    ax.imshow(x.reshape(8, 8), cmap = plt.cm.bone_r)

def get_bounds(X):
    f1_min, f1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    f2_min, f2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    return f1_min, f1_max, f2_min, f2_max


fin = open("digits.pkl", "rb")
train, test = pickle.load(fin)
X_tr, y_tr = train
X_te, y_te = test
counter = np.zeros(10)
blockarray = np.zeros(shape=(10,64))

logreg = LogisticRegression()
logreg.fit(X_tr, y_tr)
y_preds = logreg.predict(X_te)

cmatrix = confusion_matrix(y_te, y_preds)
for i in range(len(counter)):
    print("Precision score of " + str(i) + " = " + str(precision_score(y_te, y_preds, average = None)[i]))
for i in range(len(counter)):
    print("Recall score of " + str(i) + " = " + str(recall_score(y_te, y_preds, average = None)[i]))
for i in range(len(cmatrix)):
    for j in range(len(cmatrix[0])):
        if i!=j:
            print(str(cmatrix[i][j]) + " = number of predictions for " + str(i) + " that were supposed to be " + str(j))


def findbadapple(array1, array2, number):

    for i in range(len(array1)):
        if array2[i] == number:
            if array1[i] != number:

                return i
    return

def addrows(arrayone, arraytwo, rownumber, classnumber):
    for i in range(len(arrayone[0])):
        arrayone[classnumber][i] = arrayone[classnumber][i] + arraytwo[rownumber][i]
    return arrayone

def dividerows(array, rownumber, somenum):
    for i in range(len(array[0])):
        array[rownumber][i] = array[rownumber][i] / somenum
    return array

for i in range(len(y_tr)):
    counter[y_tr[i]] = counter[y_tr[i]] + 1
    blockarray = addrows(blockarray, X_tr, i, y_tr[i])

for i in range(len(blockarray)):
    blockarray = dividerows(blockarray, i, counter[i])

for i in range(10):
   print("number of " + str(i) + "'s" + " in training set is " + str(counter[i]))


for i in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 7])
    ax.set_ylim([-1, 8])
    plot_num(ax, blockarray[i, :])
    plt.show()
for i in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_num(ax, X_te[findbadapple(y_preds, y_te, i), :])
    ax.set_title("true class: " + str(i) + "classified as: " + str(y_preds[i]))
    plt.show()


fin = open("cancer.pkl", "rb")
train, test = pickle.load(fin)
cX_tr, cy_tr = train
cX_te, cy_te = test
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(cX_tr, cy_tr)
cy_preds = clf.predict(cX_te)
print("")
cancerconfusion = confusion_matrix(cy_te, cy_preds)
print(cancerconfusion)
    # your code here.
    
scaler = MinMaxScaler(feature_range=(0,1))
cX_tr = scaler.fit_transform(cX_tr)
cX_te = scaler.fit_transform(cX_te)
clf.fit(cX_tr, cy_tr)
cy_preds = clf.predict(cX_te)
cancerconfusion = confusion_matrix(cy_te, cy_preds)
print(cancerconfusion)

parametergrid = [{'C': [1,10,100,1000], 'gamma' : [1.0, 0.1, .01, .001], 'kernel' : ['rbf']}, {'C': [1,10,100,1000], 'degree': [2,3,4,5]},{'C': [1,10,100,1000], 'coef0' : [.1, 1, 10, 100]}]
clf = GridSearchCV(SVC(), parametergrid, cv=5, scoring = 'f1')
clf.fit(cX_tr, cy_tr)
print(clf.best_estimator_)
y_testpreds = clf.predict(cX_te)

bestconfusion = print("confusion matrix " + str(confusion_matrix(cy_te, y_testpreds)))

print(classification_report(cy_te, y_testpreds))




