import numpy as np
np.set_printoptions(threshold=np.nan)
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

fulltraindata = np.transpose(np.genfromtxt("posts_train.txt", unpack=True, skip_header = 1, dtype=float, delimiter=","))
fulltestdata = np.transpose(np.genfromtxt("posts_test.txt", unpack=True, skip_header = 1, dtype=float, delimiter=","))
friendsdata = np.transpose(np.genfromtxt("graph.txt", unpack=True, dtype=float, delimiter="\t"))

i = 0
IDdeletion = []

#fill null/missing ID rows with zeroes
while i < len(fulltraindata):
    print((i+1), "i")
    print(fulltraindata[i][0])
    if (fulltraindata[i][4] == 0): #if null island value
        fulltraindata[i][0] = 0 #then make ID = 0 (marked for later deletion)
        IDdeletion.append(i) #store indices of IDs to be deleted later

    if (fulltraindata[i][0] > (i + 1)):
        numinserts = int(fulltraindata[i][0] - (i+1)) #counts the number of consecutive missing ID #'s

        for j in range(numinserts):
                fulltraindata = np.insert(fulltraindata, i, 0, axis=0) #inserts a row of zeroes for the consecutive missing ID's
                i += 1
                IDdeletion.append(i) #store indices of IDs to be deleted later
    else: #if no missing ID then move onto next
        i+=1

y_tr = fulltraindata[:, [4,5]]
X_tr = fulltraindata[:, [1,2,3,6]]
X_te = fulltestdata[:, [1,2,3,4]]

scaler = MinMaxScaler(feature_range=(0,1))
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.fit_transform(X_te)

#find the IDs of all of a user's friends
def FindFriendIds(ID):

    start = np.searchsorted(friendsdata[:, 0], ID, side = 'left') #row corresponding to a user's first friend
    IDfriends = []

    while (friendsdata[start][0] == ID): #loop through all of a given user's friends
        if X_tr[int(friendsdata[start][1])-1][0] != 0: #if the friend is in the test set
            IDfriends.append(friendsdata[start][1]) #append a user's friend
        start += 1
        if start == 420582:
                break
        return IDfriends


#find median values of all of a given user's friends' features
def GetFriendsFeatures(friendarray):
    #start with empty 2d array of zeros (to be filled)
    features = np.zeros((len(friendarray), 3))
    if len(friendarray) != 0:
        for i in range(len(friendarray)):
            for j in range(3):
                if j < 2: #add latitude, longitude
                    features[i][j] = y_tr[int(friendarray[i] - 1)][j]
                else: #add most frequent posting time
                    features[i][j] = X_tr[int(friendarray[i] - 1)][0]
    else:
        subarray = [random.randint(20, 80), random.randint(20, 80), random.randint(0, 20)]
        return subarray #return random values if a user has no friends

    return np.median(features, axis = 0) #return median of all the features

#add empty columns to X train/test sets, to be filled in with friends' data
zeros = np.zeros((len(X_tr), 3))
zerostest = np.zeros((len(X_te), 3))
X_tr = np.column_stack((X_tr, zeros))
X_te = np.column_stack((X_te, zerostest))


#append friend features to both training and test sets
for i in range(len(X_tr)):
    if X_tr[i][4] != 0: #if a row has a valid user
        array = GetFriendsFeatures(FindFriendIds(i+1))
        for j in range(4,7):
            X_tr[i][j] = array[j-4] #add a user's friends' data
for i in range(len(X_te)):
    arraytest = GetFriendsFeatures(FindFriendIds(fulltestdata[i][0]))
    for j in range(3):
        X_te[i][j+4] = arraytest[j]

X_tr = [i for j, i in enumerate(X_tr) if j not in IDdeletion]
y_tr = [i for j, i in enumerate(y_tr) if j not in IDdeletion]

IDarray = [id for id in fulltestdata[:, 0]]
parametergridNN = [{'hidden_layer_sizes':[20]}]
clfNN = GridSearchCV(MLPRegressor(), parametergridNN, cv = 3, n_jobs = -1)
clfNN.fit(X_tr, y_tr)
NNypred = clfNN.predict(X_te)
np.savetxt("hello.txt", NNypred, newline = " ")

