import functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd   # Used as IO

# Reading the dataset file
data = pd.read_excel("DLdata1.xlsx" , header = None)
data = np.asarray(data)   # Convert to 2-D array
data = np.reshape(data,(250,15))
np.random.shuffle(data)

input_size = data.shape[1]-5 # Dimension of input array
rows_data  = data.shape[0]-1 # Number of rows in dataset

#normalize
for i in range(15):
    data[:,i] = functions.normalize(data[:,i])


# Splitting the dataset into inputs & targets
input_data = data[:,:input_size]  # The first three columns
targets_data = data[:,input_size]  # ،The fourth column


# Defining hyper-parameters
lr = 0.02   # Learning rate (eta)
layer = [6,3,1]    # topology including input
encoderLayers = [10 , 8 , 6  ]
epoch = 100 # Number of epochs
train_test_ratio = 0.7  # 70% of dataset used for train & 25% for test
layersLength = layer.__len__()-1


encoderOut = input_data
W = []
f, axes = plt.subplots(encoderLayers.__len__()-1, 2)
for i in range(encoderLayers.__len__()-1):
    print("####################################################################training Auto Encoder",i+1,":")
    currentLayer = [encoderLayers[i] , encoderLayers[i+1] , encoderLayers[i]]
    Wtemp , mse_train , mse_test = functions.trainAutoEncoder(encoderOut,currentLayer,lr,
                                                          epoch,train_test_ratio,rows_data)
    W.append(Wtemp[0])
    encoderOut = functions.feedForward(W,input_data)
    encoderOut = np.asarray(encoderOut)
    encoderOut = np.reshape(encoderOut , (rows_data+1 , currentLayer[1]))

    #####plot
    mse_train = np.asarray(mse_train)
    axes[i, 0].plot(mse_train)
    string = "mse train encoder layer" + str(i+1)
    axes[i, 0].set_title(string)

    mse_test = np.asarray(mse_test)
    axes[i, 1].plot(mse_test)
    string = "mse test encoder layer" + str(i+1)
    axes[i, 1].set_title(string)

    ###normalize
    for m in range(currentLayer[1]):
        encoderOut[:, m] = functions.normalize(encoderOut[:, m])
    print(np.shape(encoderOut))

plt.show()
print()
print()
print("####################################################################training LocalRecurrent:")

W , WR , mse_train , mse_test = functions.trainLocalRecurrent(encoderOut,targets_data,
                                                              layer,lr,400,train_test_ratio,rows_data)

##############################Auto Encoder


###########plot section
# plt.figure()
# plt.plot(input_data[:,0])
# plt.figure()
# plt.plot(input_data[:,1])
# plt.figure()
# plt.plot(input_data[:,2])
# plt.figure()
# plt.plot(input_data[:,3])
# plt.show()

finalLayers = encoderLayers[:encoderLayers.__len__()-1]+layer

f, axes = plt.subplots(2, 2)

mse_train = np.asarray(mse_train)
axes[0,0].plot(mse_train)
axes[0,0].set_title("mse train")

mse_test = np.asarray(mse_test)
axes[0,1].plot(mse_test)
axes[0,1].set_title("mse test")

i = input_data
feedForwardOut = functions.feedForwardLocalRecurrent(W,WR,encoderOut)
feedForwardOut = np.asarray(feedForwardOut)
feedForwardOut = feedForwardOut[:,0,0]
err = targets_data - feedForwardOut
m,b = np.polyfit(targets_data,feedForwardOut,1)
print("regression : m = ",m,"b = ",b)
axes[1,0].plot(targets_data,feedForwardOut,'bo')
axes[1,0].plot(targets_data , m*targets_data+b , 'r')
axes[1,0].set_title("regression test & train")

axes[1,1].plot(targets_data,'bo')
axes[1,1].plot(feedForwardOut , 'ro')
axes[1,1].set_title("outPut & target test & train")
plt.show()
