import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)




if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    print(f"Version of sklearn: {sklearn.__version__}")
    print("(It should be 0.20.0)")

    # your code here.


def get_favorite_data():
#here, i create a customized probability distribution on which one learner must outperform the 12 others

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

def get_lots_of_favorite_data(n, data_fun = get_favorite_data()):
    pts = [data_fun() for _ in range(n)]
    Xs, ys = zip(*pts)
    X = np.array(Xs)
    y = np.array(ys)
    return (X, y)


def plot_decision_boundary(ax, clf, x_min=0, x_max=1, y_min=0, y_max=1, res=0.01, cm=plt.cm.RdBu):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.5, cmap=cm)


def plot_num(ax, x):
    ax.imshow(x.reshape(8, 8), cmap=plt.cm.bone_r)


def get_bounds(X):
    f1_min, f1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    f2_min, f2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    return f1_min, f1_max, f2_min, f2_max


fig = [plt.figure(1, figsize = (20,6)), plt.figure(2, figsize = (20,6))]


for r in range(2): #this for loop's purpose is to open either one of two data sets
    if r==1:
        fin = open("moons.pkl", "rb")
    else:
        fin = open("simple_task.pkl", "rb")
    train, test = pickle.load(fin)
    X_tr, y_tr = train
    X_te, y_te = test
    ax = [fig[r].add_subplot(3, 5,1), fig[r].add_subplot(3,5,2), fig[r].add_subplot(3,5,3), fig[r].add_subplot(3,5,4), fig[r].add_subplot(3, 5,5)]
    ax1 = [fig[r].add_subplot(3, 5,6), fig[r].add_subplot(3,5,7), fig[r].add_subplot(3,5,8), fig[r].add_subplot(3,5,9), fig[r].add_subplot(3,5,10)]
    ax2 = [fig[r].add_subplot(3, 5, 11), fig[r].add_subplot(3,5,12), fig[r].add_subplot(3,5,13)]

    if r==0:
        for i in range(5):
            ax[i].set_ylim([-6,6])
            ax[i].set_xlim([-6, 6])
            ax1[i].set_ylim([-6, 6])
            ax1[i].set_xlim([-6, 6])
        for j in range(3):
            ax2[j].set_xlim([-6, 6])
            ax2[j].set_ylim([-6, 6])
    else:
        for i in range(5):
            ax[i].set_ylim([-3,2])
            ax[i].set_xlim([0, 3])
            ax1[i].set_ylim([-2, 3])
            ax1[i].set_xlim([-2, 2])
        for j in range(3):
            ax2[j].set_xlim([0, 3])
            ax2[j].set_ylim([-2, 3])
    # ^ setting the x and y limits of the plots

    float1, float2, float3, float4 = get_bounds(X_tr)
    for i in range(5): #train a KNN classifier with k = 1 through 5
        clf = KNeighborsClassifier(n_neighbors = i+1)
        clf.fit(X_tr, y_tr)
        ytrainpredictedKNN = clf.predict(X_tr)
        ytestpredictedKNN = clf.predict(X_te)

        if i == 4:
            clf1 = DecisionTreeClassifier("entropy")  #train a decision tree w max depth none through 4
        else:
            clf1 = DecisionTreeClassifier("entropy", max_depth=i + 1)
        clf1.fit(X_tr, y_tr)
        ytrainpredictedDT = clf1.predict(X_tr)
        ytestpredictedDT = clf1.predict(X_te)

        plot_decision_boundary(ax[i], clf, float1, float2, float3, float4)
        plot_decision_boundary(ax1[i], clf1, float1, float2, float3, float4)

        ax[i].set_xlabel("test loss " + str(zero_one_loss(ytestpredictedKNN, y_te)) + "\ntrain loss" + str(zero_one_loss(ytrainpredictedKNN, y_tr)))
        ax[i].set_title('Neighbors: %d' % (i+1))


        ax1[i].set_xlabel("test loss " + str(zero_one_loss(ytestpredictedDT, y_te)) + "\ntrain loss" + str(zero_one_loss(ytrainpredictedDT, y_tr)))

        if i == 4:
            ax1[i].set_title('None')
        else:
            ax1[i].set_title('Max Depth: %d' % (i+1))



        for k in range(len(X_te)):
            if y_te[k] > 0.5:
                ax1[i].scatter(X_te[k][0], X_te[k][1], s = 10, color = 'blue', marker = '*')
                ax[i].scatter(X_te[k][0], X_te[k][1], s=10, color='blue', marker='*')
            else:
                ax1[i].scatter(X_te[k][0], X_te[k][1], s = 10, color = 'red', marker = '*')
                ax[i].scatter(X_te[k][0], X_te[k][1], s=10, color='red', marker='*')
        for l in range(len(X_tr)):
            if y_tr[l] > 0.5:
                ax1[i].scatter(X_tr[l][0], X_tr[l][1], s  = 1,  color = 'blue')
                ax[i].scatter(X_tr[l][0], X_tr[l][1], s=1, color='blue')
            else:
                ax1[i].scatter(X_tr[l][0], X_tr[l][1], s = 1, color='red')
                ax[i].scatter(X_tr[l][0], X_tr[l][1], s=1, color='red')


    for i in range(3): #setting our own kernel function for SVM classifier

        if i == 0:
            clf2 = SVC(kernel = 'linear')
            ax2[i].set_title("linear")
        else:
            if i == 1:
                clf2 = SVC()
                ax2[i].set_title("rbf")
            else:
                clf2 = SVC(kernel = 'poly')
                ax2[i].set_title("degree 3 polynomial")
        clf2.fit(X_tr, y_tr)
        ytrainpredicted = clf2.predict(X_tr)
        ytestpredicted = clf2.predict(X_te)
        plot_decision_boundary(ax2[i], clf2, float1, float2, float3, float4)

        counter = 0
        countertrain = 0
        for j in range(len(ytestpredicted)):
            if ytestpredicted[j] != y_te[j]:
                counter += 1
            if ytrainpredicted[j] != y_tr[j]:
                countertrain += 1

        ax2[i].set_xlabel("test loss " + str(counter / len(ytestpredicted)) + "\ntrain loss" + str(countertrain / len(ytrainpredicted)))

        for k in range(len(X_te)):
            if y_te[k] > 0.5:
                ax2[i].scatter(X_te[k][0], X_te[k][1], s = 10, color = 'blue', marker = '*')

            else:
                ax2[i].scatter(X_te[k][0], X_te[k][1], s = 10, color = 'red', marker = '*')
        for l in range(len(X_tr)):
            if y_tr[l] > 0.5:
                ax2[i].scatter(X_tr[l][0], X_tr[l][1], s  = 1,  color = 'blue')
            else:
                ax2[i].scatter(X_tr[l][0], X_tr[l][1], s = 1, color='red')


fig[0].tight_layout()
fig[1].tight_layout()
plt.show() #plot the performance of all 13 learners on the two data sets


X, y = get_lots_of_favorite_data(200, get_favorite_data)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.25, random_state=42)

#testing all 13 models on customized distribution

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
            print("LAURA ") #"Laura the learner" is supposed to outperform the 12 other learners
        else:
            clf2 = SVC(kernel='poly')

    clf2.fit(X_trn, y_trn)
    ytestpredicted = clf2.predict(X_tst)
    print(zero_one_loss(y_tst, ytestpredicted))

