# Importing required packages
import numpy as np   # Used to implementing N-D matrix & tensors (Pure Python just supports 1D-arrays(lists))
import pandas as pd   # Used as IO
import math  # Used for mathematical functions or parameters
import matplotlib.pyplot as plt  # Used for plotting

# Defining hyper-parameters
lr = 0.05   # Learning rate (eta)
layer = [4,10,7,5,3]    # topology including input
epoch = 250 # Number of epochs
train_test_ratio = 0.7  # 75% of dataset used for train & 25% for test
layersLength = layer.__len__()-1
flexibleCoef = 1
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
classesNum = np.shape(classes)[0]


# find arg string
def argStr(array,x):
    for j in range(classesNum):
        if array[j] == x:
            return j
    return -1

# find confusion matrix
def findConfusion(W,input_data_train,target):
    confusionRate = np.zeros((classesNum,classesNum) , dtype=float)
    confusionNumber = np.zeros((classesNum,classesNum), dtype=int)
    count = [0, 0, 0]
    for j in range(np.shape(input_data_train)[0]):
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data_train[j, :], W[k]))
            else:
                net.append(np.dot(out[k - 1], W[k]))
            if k == W.__len__() - 1:
                out.append(suftmax(net[k]))
                m = np.argmax(out[-1])
                n = argStr(classes,target[j])
                confusionNumber[n,m]+=1
                count[n] += 1
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, layer[k + 1]))
    for i in range(classesNum):
        confusionRate[i] = confusionNumber[i] / count[i]
    return confusionNumber,confusionRate



# Defining activation functions
def sig(x):
    return 1 / (1 + np.exp(-x))

def suftmax(x):
    y = x*flexibleCoef
    denominator = 0.0
    for i in range(np.shape(y)[1]):
        denominator += np.exp(y[0,i])
    return (np.exp(y)) / denominator

def normalize(data):
    max = np.max(data)
    min = np.min(data)
    # x = (max - min)/2
    # y = max - x
    # out = data - y
    out = (data - min) / (max - min)
    return out


# Reading the dataset file
data = pd.read_csv("DLiris.csv" , header = None)
data = np.asarray(data)

#normalize
data[:,0] = normalize(data[:,0])
data[:,1] = normalize(data[:,1])
data[:,2] = normalize(data[:,2])
data[:,3] = normalize(data[:,3])

#find input size and rows
input_size = data.shape[1]-1 # Dimension of input array
rows_data =  data.shape[0]-1 # Number of rows in dataset


# Splitting data into train & test
data_train_size = round(train_test_ratio*(rows_data+1))  # 70% of dataset used for train
data_test_size = round((1 - train_test_ratio)*(rows_data+1))  # 30% of dataset used for test

data_class = []
data_train = []
data_test  = []
for i in range(classesNum):
    data_class.append(data[data[:,-1]==np.unique(data[:,-1])[i]])
    n_class = data_class[i].shape[0]
    n_train=round(train_test_ratio*n_class)
    data_train.append((data_class[i][:n_train,:]))
    data_test.append((data_class[i][n_train:,:]))

data_train = np.reshape(data_train , (np.shape(data_train)[1]*classesNum,np.shape(data_train)[2]))
data_test = np.reshape(data_test , (np.shape(data_test)[1]*classesNum , np.shape(data_test)[2]))


###################################make train data
data = data_train
np.random.shuffle(data)
input_data_train = data[:,:input_size]  # The first three columns
input_data_train = np.asarray(input_data_train , float)   # Convert to 2-D array
targets_data_train_string = data[:,input_size]  # ،The fifth column


# changing target into float
targets_data_train = np.zeros((data_train_size,classesNum) , dtype=float)
for i in range(data_train_size):
    j = argStr(classes,targets_data_train_string[i])
    targets_data_train[i,j] = 1


###################################make test data
data = data_test
np.random.shuffle(data)
input_data_test = data[:,:input_size]  # The first three columns
input_data_test = np.asarray(input_data_test , float)   # Convert to 2-D array
targets_data_test_string = data[:,input_size]  # ،The fifth column

# changing target into float
targets_data_test = np.zeros((data_test_size,classesNum) , dtype=float)
for i in range(data_test_size):
    j = argStr(classes,targets_data_test_string[i])
    targets_data_test[i,j] = 1




# Initializing weights
W = []
for i in range(layersLength):
    W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))


# Defining mean squared error for train & test
mse_train = []
mse_test = []
graph = []

# Defining net
for i in range(epoch):  # Iteration over epochs    i: number of epoch
    # np.random.shuffle(data)
    #
    # # normalize
    # data[:, 0] = normalize(data[:, 0])
    # data[:, 1] = normalize(data[:, 1])
    # data[:, 2] = normalize(data[:, 2])
    # data[:, 3] = normalize(data[:, 3])
    #
    # # Splitting the dataset into inputs & targets
    # input_data_train = data[:, :input_size]  # The first three columns
    # input_data_train = np.asarray(input_data_train, float)  # Convert to 2-D array
    # targets_data_train_string = data[:, input_size]  # ،The fifth column
    #
    # # changing target into float
    # targets_data_train = np.zeros((rows_data + 1, classesNum), dtype=float)
    # for l in range(rows_data + 1):
    #     j = argStr(classes, targets_data_train_string[l])
    #     targets_data_train[l, j] = 1

    ############################################### Train ###############################################
    err_train_epoch = 0
    for j in range(data_train_size):  # Iteration over rows   j: number of row
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data_train[j,:],W[k]))
            else:
                net.append(np.dot(out[k-1], W[k]))
            if k == W.__len__()-1:
                out.append(suftmax(net[k]))
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1 , layer[k+1]))



        #  Calculating error of the current row
        error = targets_data_train[j] - out[layersLength-1]

        gradTemp = []
        for k in range(W.__len__()):
            if k == 0:
                #np.multiply(error, temp)
                #temp = np.multiply(1-out[layersLength-1],out[layersLength-1])
                gradTemp.append(error)
            else:
                function = np.multiply(out[layersLength-k-1],1-out[layersLength-k-1])
                hassasiat = np.dot(gradTemp[k-1],W[layersLength - k].T)
                gradTemp.append(np.multiply(hassasiat,function))

        gradTemp = list(reversed(gradTemp))

        for k in range(gradTemp.__len__()):
            if k == 0:
                temp = np.reshape(input_data_train[j,:] , (np.shape(input_data_train[j,:])[0] , 1))
                gradTemp[k] = np.dot(temp,gradTemp[k])
            else:
                gradTemp[k] = np.dot(out[k-1].T,gradTemp[k])


        for k in range(W.__len__()):
            W[k] += lr * gradTemp[k]

        # ٍ Error of the row is added to error of the epoch
        tempError = 0
        for k in range(layer[-1]):
            tempError += 0.5 * error[0,k] ** 2
        err_train_epoch += tempError

    mse_train_epoch = err_train_epoch / (data_train_size)  # MSE of train for current epoch
    mse_train.append(mse_train_epoch)  # Append mse to list









    ############################################### Test ###############################################
    err_test_epoch = 0
    for j in range(data_test_size):  # Iteration over rows   j: number of row
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data_test[j, :], W[k]))
            else:
                net.append(np.dot(out[k - 1], W[k]))
            if k == W.__len__() - 1:
                out.append(suftmax(net[k]))
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, layer[k + 1]))

        error = targets_data_test[j] - out[layersLength - 1]


        # ٍ Error of the row is added to error of the epoch
        tempError = 0
        for k in range(layer[-1]):
            tempError += 0.5*error[0,k]**2
        err_test_epoch += tempError


    mse_test_epoch = err_test_epoch / (data_test_size)  # MSE of test for current epoch
    mse_test.append(mse_test_epoch)  # Append mse to list
    print('Epoch',i+1,'MSE train',mse_train_epoch,'MSE Test',mse_test_epoch)


############################################### plot section ###############################################
f, axes = plt.subplots(1, 2)


axes[0].plot(mse_train)
axes[0].set_title("mse train")


axes[1].plot(mse_test)
axes[1].set_title("mse test")

plt.show()

print("##################confusion train")
confusionNumber,confusionRate = findConfusion(W,input_data_train,targets_data_train_string)
print("##################confusion number")
print(classes)
print(confusionNumber)
print("##################confusion rate")
print(classes)
print(confusionRate)

print("##################confusion test")
confusionNumber,confusionRate = findConfusion(W,input_data_test,targets_data_test_string)
print("##################confusion number")
print(classes)
print(confusionNumber)
print("##################confusion rate")
print(classes)
print(confusionRate)

