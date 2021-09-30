# Importing required packages
import numpy as np   # Used to implementing N-D matrix & tensors (Pure Python just supports 1D-arrays(lists))
import pandas as pd   # Used as IO
import math  # Used for mathematical functions or parameters
import matplotlib.pyplot as plt  # Used for plotting

# feed forward function
def feedForward(W,WL,WU,input_data):
    outList = []
    for j in range(np.shape(input_data)[0]):
        net = []
        out = []
        outU = []
        outL = []
        netU = []
        netL = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j, :], W[k]))
                out.append(sig(net[k]))
            elif k == W.__len__() - 1:
                netU.append(np.dot(out[k - 1], WU[0]))
                netL.append(np.dot(out[k - 1], WL[0]))
                outU.append(netU[0])
                outL.append(netL[0])
                out.append((netU[0] + netL[0]) / 2)
                outList.append(out[k])
            else:
                net.append(np.dot(out[k - 1], W[k]))
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, layer[k + 1]))

    outList = np.reshape(outList,(56))
    return outList

# Defining activation functions
def sig(x):
    return 1 / (1 + math.e**-x)

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
#np.random.shuffle(data)

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
q
epoch = 400 # Number of epochs
train_test_ratio = 0.7  # 70% of dataset used for train & 25% for test
layersLength = layer.__len__()-1




# Splitting data into train & test
data_train_size = round(train_test_ratio*rows_data)  # 70% of dataset used for train
data_test_size = round((1 - train_test_ratio)*rows_data)  # 30% of dataset used for test




# Initializing weights
W = []
WU = []
WL = []

for i in range(layersLength):
    W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))

WU.append(np.random.uniform(-1, 1, (layer[layersLength-1], layer[layersLength])))
WL.append(np.random.uniform(-1, 1, (layer[layersLength-1], layer[layersLength])))

# Defining mean squared error for train & test
mse_train = []
mse_test = []

# Defining net
for i in range(epoch):  # Iteration over epochs    i: number of epoch

    ############################################### Train ###############################################
    err_train_epoch = 0
    for j in range(data_train_size):  # Iteration over rows   j: number of row
        net = []
        out = []
        outU = []
        outL = []
        netU = []
        netL = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j,:],W[k]))
                out.append(sig(net[k]))
            elif k == W.__len__()-1:
                netU.append(np.dot(out[k-1], WU[0]))
                netL.append(np.dot(out[k - 1], WL[0]))
                outU.append(netU[0])
                outL.append(netL[0])
                out.append( (netU[0]+netL[0]) / 2)
            else:
                net.append(np.dot(out[k - 1], W[k]))
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1 , layer[k+1]))

        #  Calculating error of the current row
        error = targets_data[j] - out[layersLength-1]


        #Hidden layer
        gradTemp = []

        for k in range(W.__len__()):
            if k == 0:
                gradTemp.append(error / 2)
            elif k == 1:
                function = np.multiply(out[layersLength - k - 1], 1 - out[layersLength - k - 1])
                hassasiat = np.dot(gradTemp[k - 1], WL[0].T) + np.dot(gradTemp[k - 1], WU[0].T)
                gradTemp.append(np.multiply(hassasiat, function))
            else:
                function = np.multiply(out[layersLength - k - 1], 1 - out[layersLength - k - 1])
                hassasiat = np.dot(gradTemp[k - 1], W[layersLength - k].T)
                gradTemp.append(np.multiply(hassasiat, function))


        gradTemp = list(reversed(gradTemp))

        for k in range(gradTemp.__len__()):
            if k == 0:
                temp = np.reshape(input_data[j, :], (np.shape(input_data[j, :])[0], 1))
                gradTemp[k] = np.dot(temp, gradTemp[k])
            else:
                gradTemp[k] = np.dot(out[k-1].T,gradTemp[k])


        for k in range(W.__len__()-1):
            W[k] += lr * gradTemp[k]

        WU[0]  += lr * gradTemp[W.__len__()-1]
        WL[0]  += lr * gradTemp[W.__len__()-1]


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
        net = []
        out = []
        outU = []
        outL = []
        netU = []
        netL = []
        index = j + data_train_size + 1
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[index, :], W[k]))
                out.append(sig(net[k]))
            elif k == W.__len__() - 1:
                netU.append(np.dot(out[k - 1], WU[0]))
                netL.append(np.dot(out[k - 1], WL[0]))
                outU.append(netU[0])
                outL.append(netL[0])
                out.append((netU[0] + netL[0]) / 2)
            else:
                net.append(np.dot(out[k - 1], W[k]))
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, layer[k + 1]))

        error = targets_data[index] - out[layersLength - 1]


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
feedForwardOut = feedForward(W,WL,WU,input_data)
m,b = np.polyfit(targets_data,feedForwardOut,1)
print("regression : m = ",m,"b = ",b)
axes[1,0].plot(targets_data,feedForwardOut,'bo')
axes[1,0].plot(targets_data , m*targets_data+b , 'r')
axes[1,0].set_title("regression test & train")

axes[1,1].plot(targets_data,'bo')
axes[1,1].plot(feedForwardOut , 'ro')
axes[1,1].set_title("outPut & target test & train")
plt.show()


