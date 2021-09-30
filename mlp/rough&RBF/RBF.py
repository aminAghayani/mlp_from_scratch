# Importing required packages
import numpy as np   # Used to implementing N-D matrix & tensors (Pure Python just supports 1D-arrays(lists))
import pandas as pd   # Used as IO
import math  # Used for mathematical functions or parameters
import matplotlib.pyplot as plt  # Used for plotting

# feed forward function
def feedForward(W,C,SIGMA,input_data,layer):
    outList = []
    for j in range(np.shape(input_data)[0]):
        net = np.zeros(layer[1])

        for k in range(layer[1]):
            for m in range(layer[0]):
                net[k] += np.power(input_data[j, m] - C[k, m], 2)
        net = np.sqrt(net)
        out1 = guassian(net, SIGMA)
        out2 = np.dot(out1, W)
        outList.append(out2)
    outList = np.reshape(outList,(56))
    return outList

# Defining activation functions
def sig(x):
    return 1 / (1 + math.e**-x)

def guassian(net,SIGMA):
    out = np.power(net/SIGMA,2)
    return np.exp(-0.5*out)

def normalize(data):
    max = np.max(data)
    min = np.min(data)
    x = (max - min)/2
    y = max - x
    out = data - y
    out = out / x
    return out


# Reading the dataset file
data = pd.read_excel("DLdata2.xlsx" , header = None)
data = np.asarray(data)   # Convert to 2-D array
data = normalize(data)
data = np.reshape(data,(56,6))
dataCopy = data.copy()
np.random.shuffle(data)

input_size = data.shape[1]-1 # Dimension of input array
rows_data  = data.shape[0]-1 # Number of rows in dataset

#normalize
for i in range(6):
    data[:,i] = normalize(data[:,i])



# Splitting the dataset into inputs & targets
input_data = data[:,:input_size]  # The first three columns
targets_data = data[:,input_size]  # ،The fourth column


# Defining hyper-parameters
lr = 0.02   # Learning rate (eta)
layer = [5,25,1]    # topology including input
epoch = 30 # Number of epochs
train_test_ratio = 0.7  # 70% of dataset used for train & 25% for test
layersLength = layer.__len__()-1




# Splitting data into train & test
data_train_size = round(train_test_ratio*rows_data)  # 70% of dataset used for train
data_test_size = round((1 - train_test_ratio)*rows_data)  # 30% of dataset used for test




# Initializing free variables
W = []
C = []
SIGMA = []

W = np.random.uniform(-1, 1, (layer[1], layer[2]))
C = np.random.uniform(-1, 1, (layer[1] , layer[0]))
SIGMA = np.random.uniform(1, 5, (1, layer[1]))

# Defining mean squared error for train & test
mse_train = []
mse_test = []

# Defining net
for i in range(epoch):  # Iteration over epochs    i: number of epoch
    ############################################### Train ###############################################
    err_train_epoch = 0
    for j in range(data_train_size):  # Iteration over rows   j: number of row
        net = np.zeros(layer[1])

        for k in range(layer[1]):
            for m in range(layer[0]):
                net[k] += np.power(input_data[j,m]-C[k,m],2)
        net = np.sqrt(net)
        out1 = guassian(net,SIGMA)
        out2 = np.dot(out1,W)

        error = targets_data[j] - out2

        ##grad W
        gradW = np.dot(error,out1)

        ##grad C
        temp = np.zeros((layer[1],layer[0]))
        for k in range(layer[1]):
            for m in range(layer[0]):
                temp[k,m] += ( (input_data[j,m]-C[k,m]) / np.power(SIGMA[0,k],2) )
        gradC = np.dot(error,W.T)
        gradC = np.dot(gradC,temp)
        gradC = np.dot(out1.T,gradC)

        ## grad SIGMA
        temp = np.power(net,2) / np.power(SIGMA[0],3)
        gradSIGMA = np.dot(error,W.T)
        gradSIGMA = np.dot(gradSIGMA,temp.T)
        gradSIGMA = np.dot(gradSIGMA,out1)

        ###update
        W += lr*gradW.T
        C += lr*gradC
        SIGMA += lr*gradSIGMA

        # ٍ Error of the row is added to error of the epoch
        tempError = 0
        for k in range(layer[-1]):
            tempError += 0.5 * error[0, k] ** 2
        err_train_epoch += tempError

    mse_train_epoch = err_train_epoch / data_train_size  # MSE of train for current epoch
    mse_train.append(mse_train_epoch)  # Append mse to list






    ############################################### Test ###############################################
    err_test_epoch = 0
    for j in range(data_test_size):  # Iteration over rows   j: number of row
        net = np.zeros(layer[1])
        index = j + data_train_size + 1

        for k in range(layer[1]):
            for m in range(layer[0]):
                net[k] += np.power(input_data[index, m] - C[k, m], 2)
        net = np.sqrt(net)
        out1 = guassian(net, SIGMA)
        out2 = np.dot(out1, W)

        error = targets_data[index] - out2


        # ٍ Error of the row is added to error of the epoch
        tempError = 0
        for k in range(layer[-1]):
            tempError += 0.5 * error[0, k] ** 2
        err_test_epoch += tempError


    mse_test_epoch = err_test_epoch / data_test_size  # MSE of test for current epoch
    mse_test.append(mse_test_epoch)  # Append mse to list
    print('Epoch',i+1,'MSE train',mse_train_epoch,'MSE Test',mse_test_epoch)


###########plot section
# plt.figure()
# plt.plot(graph)
# plt.plot(input_data[:,0])
# plt.figure()
# plt.plot(input_data[:,1])
# plt.figure()
# plt.plot(input_data[:,2])
# plt.figure()
# plt.plot(input_data[:,3])
# plt.show()

f, axes = plt.subplots(2, 2)

mse_train = np.asarray(mse_train)
axes[0,0].plot(mse_train)
axes[0,0].set_title("mse train")

mse_test = np.asarray(mse_test)
axes[0,1].plot(mse_test)
axes[0,1].set_title("mse test")

i = input_data
o = dataCopy[:,input_size]
feedForwardOut = feedForward(W,C,SIGMA,dataCopy[:,:input_size],layer)
m,b = np.polyfit(o,feedForwardOut,1)
print("regression : m = ",m,"b = ",b)
axes[1,0].plot(o,feedForwardOut,'bo')
axes[1,0].plot(targets_data , m*targets_data+b , 'r')
axes[1,0].set_title("regression test & train")

axes[1,1].plot(o,'bo')
axes[1,1].plot(feedForwardOut , 'ro')
axes[1,1].set_title("outPut & target test & train")
plt.show()


