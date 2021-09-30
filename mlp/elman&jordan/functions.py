# Importing required packages
import numpy as np   # Used to implementing N-D matrix & tensors (Pure Python just supports 1D-arrays(lists))
import math  # Used for mathematical functions or parameters



# feed forward functions
def feedForward(W,input_data):
    outList = []
    rowData = np.shape(input_data)[0]
    for j in range(rowData):
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j, :], W[k]))
            else:
                net.append(np.dot(out[k - 1], W[k]))
            if k == W.__len__() - 1:
                out.append(net[k])
                outList.append(net[k])
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, np.shape(W[k])[1]))
    return outList

def feedForwardLocalRecurrent(W,WR,input_data):
    outList = []
    rowData = np.shape(input_data)[0]
    lastOut = []
    for i in range(W.__len__()):
        lastOut.append(np.zeros( (1 , np.shape(W[i])[1]) ))
        print("last out" , np.shape(lastOut[i]))

    for j in range(rowData):
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOut[k], WR[k]))
            else:
                net.append(np.dot(out[k - 1], W[k]) + np.dot(lastOut[k], WR[k]))
            if k == W.__len__() - 1:
                out.append(net[k])
                outList.append(net[k])
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, np.shape(W[k])[1]))

        for k in range(W.__len__()):
            lastOut[k] = out[k]

    return outList

def feedForwardElman(W,WR,input_data):
    outList = []
    rowData = np.shape(input_data)[0]
    lastOut = []
    lastOut.append(np.zeros( (1 , np.shape(W[0])[1]) ))
    print("last out" , np.shape(lastOut[0]))

    for j in range(rowData):
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOut[0], WR[0]))
            else:
                net.append(np.dot(out[k - 1], W[k]))
            if k == W.__len__() - 1:
                out.append(net[k])
                outList.append(net[k])
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, np.shape(W[k])[1]))


        lastOut[0] = out[0]

    return outList

def feedForwardJordan(W,WR,input_data):
    outList = []
    rowData = np.shape(input_data)[0]
    lastOut = []
    lastOut.append(np.zeros( (1 , np.shape(W[1])[1]) ))
    print("last out" , np.shape(lastOut[0]))

    for j in range(rowData):
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOut[0], WR[0]))
            else:
                net.append(np.dot(out[k - 1], W[k]))
            if k == W.__len__() - 1:
                out.append(net[k])
                outList.append(net[k])
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, np.shape(W[k])[1]))

        lastOut[0] = out[1]

    return outList

def feedForwardElmanJordan(W, WElman , WJordan , input_data):
    outList = []
    rowData = np.shape(input_data)[0]
    lastOutElman = []
    lastOutJordan = []
    lastOutElman.append(np.zeros((1, np.shape(W[0])[1])))
    print("last out", np.shape(lastOutElman[0]))
    lastOutJordan.append(np.zeros((1, np.shape(W[1])[1])))
    print("last out", np.shape(lastOutJordan[0]))

    for j in range(rowData):
        net = []
        out = []
        for k in range(W.__len__()):
            if k == 0:
                net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOutElman[k], WElman[k])
                           + np.dot(lastOutJordan[k], WJordan[k]))
            else:
                net.append(np.dot(out[k - 1], W[k]))
            if k == W.__len__() - 1:
                out.append(net[k])
                outList.append(net[k])
            else:
                out.append(sig(net[k]))
            out[k] = np.reshape(out[k], (1, np.shape(W[k])[1]))

        lastOutElman[0] = out[0]
        lastOutJordan[0] = out[1]

    return outList

# Defining activation functions
def sig(x):
    return 1 / (1 + math.e**(-x))

# Normalize
def normalize(data):
    max = np.max(data)
    min = np.min(data)
    x = (max - min)
    return (data - min) / x

# Training functions
def trainPerceptron(input_data,targets_data,layer,lr,epoch,train_test_ratio,rows_data):

    layersLength = layer.__len__() - 1

    # Splitting data into train & test
    data_train_size = round(train_test_ratio * rows_data)  # 70% of dataset used for train
    data_test_size = round((1 - train_test_ratio) * rows_data)  # 30% of dataset used for test

    # Initializing weights
    W = []
    for i in range(layersLength):
        W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))

    # Defining mean squared error for train & test
    mse_train = []
    mse_test  = []

    # Defining net
    for i in range(epoch):  # Iteration over epochs    i: number of epoch
        ############################################### Train ###############################################
        err_train_epoch = 0
        for j in range(data_train_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            #  Calculating error of the current row
            error = targets_data[j] - out[layersLength - 1]

            # Hidden layer
            gradTemp = []

            for k in range(W.__len__()):
                if k == 0:
                    gradTemp.append(error)
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

            for k in range(W.__len__()):
                W[k] += lr * gradTemp[k]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_train_epoch += tempError

        mse_train_epoch = err_train_epoch / (rows_data + 1)  # MSE of train for current epoch
        mse_train.append(mse_train_epoch)  # Append mse to list

        ############################################### Test ###############################################
        err_test_epoch = 0
        for j in range(data_test_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            error = targets_data[j] - out[layersLength - 1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_test_epoch += tempError

        mse_test_epoch = err_test_epoch / (rows_data + 1)  # MSE of test for current epoch
        mse_test.append(mse_test_epoch)  # Append mse to list
        print('Epoch', i + 1, 'MSE train', mse_train_epoch, 'MSE Test', mse_test_epoch)

    return W,mse_train,mse_test

def trainAutoEncoder(input_data,layer,lr,epoch,train_test_ratio,rows_data):

    targets_data = input_data
    layersLength = layer.__len__() - 1

    # Splitting data into train & test
    data_train_size = round(train_test_ratio * rows_data)  # 70% of dataset used for train
    data_test_size = round((1 - train_test_ratio) * rows_data)  # 30% of dataset used for test

    # Initializing weights
    W = []
    for i in range(layersLength):
        W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))


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
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j,:],W[k]))
                else:
                    net.append(np.dot(out[k-1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1 , layer[k+1]))

            #  Calculating error of the current row
            error = targets_data[j] - out[layersLength-1]


            #Hidden layer
            gradTemp = []

            for k in range(W.__len__()):
                if k == 0:
                    gradTemp.append(error)
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
                    gradTemp[k] = np.dot(out[k - 1].T, gradTemp[k])


            W[1] += lr * gradTemp[1]
            W[0] = W[1].T

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_train_epoch += tempError / layer[-1]

        mse_train_epoch = err_train_epoch / (rows_data+1)  # MSE of train for current epoch
        mse_train.append(mse_train_epoch)  # Append mse to list







        ############################################### Test ###############################################
        err_test_epoch = 0
        for j in range(data_test_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))


            # calculate error
            error = targets_data[j] - out[layersLength - 1]


            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_test_epoch += tempError / layer[-1]


        mse_test_epoch = err_test_epoch / (rows_data+1)  # MSE of test for current epoch
        mse_test.append(mse_test_epoch)  # Append mse to list
        print('Epoch',i+1,'MSE train',mse_train_epoch,'MSE Test',mse_test_epoch)

    return W,mse_train,mse_test

def trainLocalRecurrent(input_data,targets_data,layer,lr,epoch,train_test_ratio,rows_data):

    layersLength = layer.__len__() - 1

    # Splitting data into train & test
    data_train_size = round(train_test_ratio * rows_data)  # 70% of dataset used for train
    data_test_size = round((1 - train_test_ratio) * rows_data)  # 30% of dataset used for test

    # Initializing weights
    W  = []
    WR = []
    lastOut = []
    for i in range(layersLength):
        W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))

    for i in range(layersLength):
        WR.append(np.random.uniform(-1, 1, (layer[i+1], layer[i+1])))
        print("WR shape : " , np.shape(WR[i]))

    for i in range(layersLength):
        lastOut.append(np.zeros( (1 , layer[i+1]) ))
        print("last out" , np.shape(lastOut[i]))


    # Defining mean squared error for train & test
    mse_train = []
    mse_test  = []

    # Defining net
    for i in range(epoch):  # Iteration over epochs    i: number of epoch
        ############################################### Train ###############################################
        err_train_epoch = 0
        for j in range(data_train_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOut[k], WR[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]) + np.dot(lastOut[k], WR[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            #  Calculating error of the current row
            error = targets_data[j] - out[layersLength - 1]

            # Hidden layer
            gradTemp = []
            gradRTemp = []

            for k in range(W.__len__()):
                if k == 0:
                    gradTemp.append(error)
                else:
                    function = np.multiply(out[layersLength - k - 1], 1 - out[layersLength - k - 1])
                    hassasiat = np.dot(gradTemp[k - 1], W[layersLength - k].T)
                    gradTemp.append(np.multiply(hassasiat, function))

            gradTemp = list(reversed(gradTemp))

            for k in range(gradTemp.__len__()):
                gradRTemp.append(np.dot(lastOut[k].T , gradTemp[k]))
                if k == 0:
                    temp = np.reshape(input_data[j, :], (np.shape(input_data[j, :])[0], 1))
                    gradTemp[k] = np.dot(temp, gradTemp[k])
                else:
                    gradTemp[k] = np.dot(out[k-1].T,gradTemp[k])

            for k in range(W.__len__()):
                W[k] += lr * gradTemp[k]
                WR[k] += lr * gradRTemp[k]

            for k in range(W.__len__()):
                lastOut[k] = out[k]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_train_epoch += tempError / layer[-1]

        mse_train_epoch = err_train_epoch / (data_train_size + 1)  # MSE of train for current epoch
        mse_train.append(mse_train_epoch)  # Append mse to list

        ############################################### Test ###############################################
        err_test_epoch = 0
        for j in range(data_test_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            index = j + data_train_size + 1
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[index, :], W[k]) + np.dot(lastOut[k], WR[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]) + np.dot(lastOut[k], WR[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            for k in range(W.__len__()):
                lastOut[k] = out[k]

            error = targets_data[index] - out[layersLength - 1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_test_epoch += tempError  / layer[-1]

        mse_test_epoch = err_test_epoch / (data_test_size + 1)  # MSE of test for current epoch
        mse_test.append(mse_test_epoch)  # Append mse to list
        print('Epoch', i + 1, 'MSE train', mse_train_epoch, 'MSE Test', mse_test_epoch)

    return W,WR,mse_train,mse_test

def trainElman(input_data,targets_data,layer,lr,epoch,train_test_ratio,rows_data):

    layersLength = layer.__len__() - 1

    # Splitting data into train & test
    data_train_size = round(train_test_ratio * rows_data)  # 70% of dataset used for train
    data_test_size = round((1 - train_test_ratio) * rows_data)  # 30% of dataset used for test

    # Initializing weights
    W  = []
    WR = []
    lastOut = []
    for i in range(layersLength):
        W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))

        WR.append(np.random.uniform(-1, 1, (layer[1], layer[1])))
        print("WR shape : " , np.shape(WR[0]))

        lastOut.append(np.zeros( (1 , layer[1]) ))
        print("last out" , np.shape(lastOut[0]))


    # Defining mean squared error for train & test
    mse_train = []
    mse_test  = []

    # Defining net
    for i in range(epoch):  # Iteration over epochs    i: number of epoch
        ############################################### Train ###############################################
        err_train_epoch = 0
        for j in range(data_train_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOut[0], WR[0]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            #  Calculating error of the current row
            error = targets_data[j] - out[layersLength - 1]

            # Hidden layer
            gradTemp = []
            gradRTemp = []

            for k in range(W.__len__()):
                if k == 0:
                    gradTemp.append(error)
                else:
                    function = np.multiply(out[layersLength - k - 1], 1 - out[layersLength - k - 1])
                    hassasiat = np.dot(gradTemp[k - 1], W[layersLength - k].T)
                    gradTemp.append(np.multiply(hassasiat, function))

            gradTemp = list(reversed(gradTemp))

            for k in range(gradTemp.__len__()):
                if k == 0:
                    gradRTemp.append(np.dot(lastOut[k].T, gradTemp[k]))
                    temp = np.reshape(input_data[j, :], (np.shape(input_data[j, :])[0], 1))
                    gradTemp[k] = np.dot(temp, gradTemp[k])
                else:
                    gradTemp[k] = np.dot(out[k-1].T,gradTemp[k])

            for k in range(W.__len__()):
                W[k] += lr * gradTemp[k]

            WR[0] += lr * gradRTemp[0]
            lastOut[0] = out[0]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_train_epoch += tempError / layer[-1]

        mse_train_epoch = err_train_epoch / (data_train_size + 1)  # MSE of train for current epoch
        mse_train.append(mse_train_epoch)  # Append mse to list

        ############################################### Test ###############################################
        err_test_epoch = 0
        for j in range(data_test_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            index = j + data_train_size + 1
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[index, :], W[k]) + np.dot(lastOut[0], WR[0]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))


            lastOut[0] = out[0]

            error = targets_data[index] - out[layersLength - 1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_test_epoch += tempError  / layer[-1]

        mse_test_epoch = err_test_epoch / (data_test_size + 1)  # MSE of test for current epoch
        mse_test.append(mse_test_epoch)  # Append mse to list
        print('Epoch', i + 1, 'MSE train', mse_train_epoch, 'MSE Test', mse_test_epoch)

    return W,WR,mse_train,mse_test

def trainJordan(input_data,targets_data,layer,lr,epoch,train_test_ratio,rows_data):

    layersLength = layer.__len__() - 1

    # Splitting data into train & test
    data_train_size = round(train_test_ratio * rows_data)  # 70% of dataset used for train
    data_test_size = round((1 - train_test_ratio) * rows_data)  # 30% of dataset used for test

    # Initializing weights
    W  = []
    WR = []
    lastOut = []
    for i in range(layersLength):
        W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))

    WR.append(np.random.uniform(-1, 1, (layer[2], layer[1])))
    print("WR shape : " , np.shape(WR[0]))

    lastOut.append(np.zeros( (1 , layer[2]) ))
    print("last out" , np.shape(lastOut[0]))


    # Defining mean squared error for train & test
    mse_train = []
    mse_test  = []

    # Defining net
    for i in range(epoch):  # Iteration over epochs    i: number of epoch
        ############################################### Train ###############################################
        err_train_epoch = 0
        for j in range(data_train_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOut[0], WR[0]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            #  Calculating error of the current row
            error = targets_data[j] - out[layersLength - 1]

            # Hidden layer
            gradTemp = []
            gradRTemp = []

            for k in range(W.__len__()):
                if k == 0:
                    gradTemp.append(error)
                else:
                    function = np.multiply(out[layersLength - k - 1], 1 - out[layersLength - k - 1])
                    hassasiat = np.dot(gradTemp[k - 1], W[layersLength - k].T)
                    gradTemp.append(np.multiply(hassasiat, function))

            gradTemp = list(reversed(gradTemp))

            for k in range(gradTemp.__len__()):
                if k == 0:
                    gradRTemp.append(np.dot(lastOut[0].T, gradTemp[0]))
                    temp = np.reshape(input_data[j, :], (np.shape(input_data[j, :])[0], 1))
                    gradTemp[k] = np.dot(temp, gradTemp[k])
                else:
                    gradTemp[k] = np.dot(out[k-1].T,gradTemp[k])

            for k in range(W.__len__()):
                W[k] += lr * gradTemp[k]

            WR[0] += lr * gradRTemp[0]

            lastOut[0] = out[1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_train_epoch += tempError / layer[-1]

        mse_train_epoch = err_train_epoch / (data_train_size + 1)  # MSE of train for current epoch
        mse_train.append(mse_train_epoch)  # Append mse to list

        ############################################### Test ###############################################
        err_test_epoch = 0
        for j in range(data_test_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            index = j + data_train_size + 1
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[index, :], W[k]) + np.dot(lastOut[0], WR[0]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))


            lastOut[0] = out[1]

            error = targets_data[index] - out[layersLength - 1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_test_epoch += tempError  / layer[-1]

        mse_test_epoch = err_test_epoch / (data_test_size + 1)  # MSE of test for current epoch
        mse_test.append(mse_test_epoch)  # Append mse to list
        print('Epoch', i + 1, 'MSE train', mse_train_epoch, 'MSE Test', mse_test_epoch)

    return W,WR,mse_train,mse_test

def trainElmanJordan(input_data,targets_data,layer,lr,epoch,train_test_ratio,rows_data):

    layersLength = layer.__len__() - 1

    # Splitting data into train & test
    data_train_size = round(train_test_ratio * rows_data)  # 70% of dataset used for train
    data_test_size = round((1 - train_test_ratio) * rows_data)  # 30% of dataset used for test

    # Initializing weights
    W  = []
    WElman = []
    WJordan = []
    lastOutElman = []
    lastOutJordan = []
    for i in range(layersLength):
        W.append(np.random.uniform(-1, 1, (layer[i], layer[i+1])))

    WElman.append(np.random.uniform(-1, 1, (layer[1], layer[1])))
    print("WR shape : " , np.shape(WElman[0]))
    WJordan.append(np.random.uniform(-1, 1, (layer[2], layer[1])))
    print("WR shape : ", np.shape(WJordan[0]))

    lastOutElman.append(np.zeros( (1 , layer[1]) ))
    print("last out" , np.shape(lastOutElman[0]))
    lastOutJordan.append(np.zeros((1, layer[2])))
    print("last out", np.shape(lastOutJordan[0]))


    # Defining mean squared error for train & test
    mse_train = []
    mse_test  = []

    # Defining net
    for i in range(epoch):  # Iteration over epochs    i: number of epoch
        ############################################### Train ###############################################
        err_train_epoch = 0
        for j in range(data_train_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[j, :], W[k]) + np.dot(lastOutElman[k], WElman[k])
                               + np.dot(lastOutJordan[k], WJordan[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            #  Calculating error of the current row
            error = targets_data[j] - out[layersLength - 1]

            # Hidden layer
            gradTemp = []
            gradElmanTemp = []
            gradJordanTemp = []

            for k in range(W.__len__()):
                if k == 0:
                    gradTemp.append(error)
                else:
                    function = np.multiply(out[layersLength - k - 1], 1 - out[layersLength - k - 1])
                    hassasiat = np.dot(gradTemp[k - 1], W[layersLength - k].T)
                    gradTemp.append(np.multiply(hassasiat, function))

            gradTemp = list(reversed(gradTemp))

            for k in range(gradTemp.__len__()):
                if k == 0:
                    gradElmanTemp.append(np.dot(lastOutElman[k].T, gradTemp[k]))
                    gradJordanTemp.append(np.dot(lastOutJordan[k].T, gradTemp[k]))
                    temp = np.reshape(input_data[j, :], (np.shape(input_data[j, :])[0], 1))
                    gradTemp[k] = np.dot(temp, gradTemp[k])
                else:
                    gradTemp[k] = np.dot(out[k-1].T,gradTemp[k])

            for k in range(W.__len__()):
                W[k] += lr * gradTemp[k]

            WElman[0] += lr * gradElmanTemp[0]
            WJordan[0] += lr * gradJordanTemp[0]

            lastOutElman[0] = out[0]
            lastOutJordan[0] = out[1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_train_epoch += tempError / layer[-1]

        mse_train_epoch = err_train_epoch / (data_train_size + 1)  # MSE of train for current epoch
        mse_train.append(mse_train_epoch)  # Append mse to list

        ############################################### Test ###############################################
        err_test_epoch = 0
        for j in range(data_test_size):  # Iteration over rows   j: number of row
            net = []
            out = []
            index = j + data_train_size + 1
            for k in range(W.__len__()):
                if k == 0:
                    net.append(np.dot(input_data[index, :], W[k]) + np.dot(lastOutElman[k], WElman[k])
                               + np.dot(lastOutJordan[k], WJordan[k]))
                else:
                    net.append(np.dot(out[k - 1], W[k]))
                if k == W.__len__() - 1:
                    out.append(net[k])
                else:
                    out.append(sig(net[k]))
                out[k] = np.reshape(out[k], (1, layer[k + 1]))

            lastOutElman[0] = out[0]
            lastOutJordan[0] = out[1]

            error = targets_data[index] - out[layersLength - 1]

            # ٍ Error of the row is added to error of the epoch
            tempError = 0
            for k in range(layer[-1]):
                tempError += 0.5 * error[0, k] ** 2
            err_test_epoch += tempError  / layer[-1]

        mse_test_epoch = err_test_epoch / (data_test_size + 1)  # MSE of test for current epoch
        mse_test.append(mse_test_epoch)  # Append mse to list
        print('Epoch', i + 1, 'MSE train', mse_train_epoch, 'MSE Test', mse_test_epoch)

    return W,WElman,WJordan,mse_train,mse_test
