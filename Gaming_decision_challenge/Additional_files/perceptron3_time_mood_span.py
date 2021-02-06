import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt



def read (name):
    data = []
    with open(name, 'r') as f:
        for line in f:
            data.append( line.strip().split(',') )

    train_data = np.zeros((len(data), 15))
    #train_tag = np.zeros((len(data), 1))
    train_tag = np.zeros(len(data))
    for i in range( len(data)):
        train_data[i][0] =1

        if data[i][0].lower() == "weekday":
            train_data[i][1] = 1
        elif data[i][0].lower() == "saturday":
            train_data[i][1] = 2
        elif data[i][0].lower() == "sunday":
            train_data[i][1] = 3

        if data[i][1].lower() == "morning":
            train_data[i][2] = 1
        elif data[i][1].lower() == "afternoon":
            train_data[i][3] = 1
        elif data[i][1].lower() == "evening":
            train_data[i][4] = 1

        if data[i][2] == "<30":
            train_data[i][5] = 1
        elif data[i][2] == "30-60":
            train_data[i][6] = 1
        elif data[i][2] == ">60":
            train_data[i][7] = 1

        if data[i][3].lower() == "silly":
            train_data[i][8] = 1
        elif data[i][3].lower() == "happy":
            train_data[i][9] = 1
        elif data[i][3].lower() == "tired":
            train_data[i][10] = 1

        if data[i][4].lower() == "yes":
            train_data[i][11] = 1
        if data[i][5].lower() == "yes":
            train_data[i][12] = 1
        if data[i][6].lower() == "yes":
            train_data[i][13] = 1
        if data[i][7].lower() == "yes":
            train_data[i][14] = 1

        if data[i][8].lower() == "settersofcatan":
            train_tag[i] = 1
        elif data[i][8].lower() == "applestoapples":
            train_tag[i] = 0

    return data,train_data,train_tag

def train(L, train_data, train_tag, test_data, test_tag):


    acc_normal = []
    acc_ave = []

    for epoch in L:
        print("Starting Epoch %s" % epoch)
        w = np.zeros((1, np.shape(train_data)[1]))[0]
        t = np.zeros((1, np.shape(train_data)[1]))[0]
        c=1
        for num in range(epoch):

            for i in range(np.shape(train_data)[0]):
                if train_data[i].dot(w) >=0:
                    h = 1
                else:
                    h=0
                w = w + (train_tag[i] - h) * train_data[i]
                # t = w+ ( (train_tag[i] - h) * train_data[i] ) *c
                t = t + w
                #c += 1
                # print(w)

        t = 1/( epoch * np.shape(train_data)[0])*t.copy()
        print(t)
        #t = w- 1/c*t


        # y_train_normal = np.matmul(train_data,w)
        # y_train_ave = np.matmul(train_data,t)
        #
        # y_test_normal = np.matmul(test_data,w)
        # y_test_ave = np.matmul (test_data,t)

        y_train_normal = feedforward(train_data,w)
        y_train_ave = feedforward(train_data,t)

        y_test_normal = feedforward(test_data,w)
        y_test_ave = feedforward (test_data,t)

        acc_normal.append( (epoch, accuracy(y_train_normal,train_tag) , accuracy(y_test_normal,test_tag) ) )
        acc_ave.append( (epoch, accuracy(y_train_ave,train_tag) , accuracy(y_test_ave,test_tag) ) )


    return acc_normal, acc_ave

def accuracy (y_model, y_tag):

    count = 0
    for i in range(np.shape(y_tag)[0] ):
        if y_model[i] == y_tag[i]:
            count +=1

    return (count / np.shape(y_tag)[0])


def feedforward(data, weight):

    output = np.matmul(data,weight)

    for i in range(np.shape(output)[0]):
        if output[i]>=0:
            output[i]=1
        else:
            output[i]=0

    return output






if __name__=="__main__":

    #f = np.loadtxt("data/game_attrdata_train.dat", delimiter=",")

    name_train = "data/game_attrdata_train.dat"
    name_test =  "data/game_attrdata_test.dat"

    data_tr, train_data, train_tag = read(name_train)
    data_ts, test_data, test_tag = read(name_test)


    #L = np.arange(100,2100,100)
    L = np.arange(0,600,10)


    #L = np.arange(2000, 4100, 1000)
    acc_normal, acc_ave = train (L, train_data,train_tag, test_data, test_tag)


    ep = [a for (a,b,c) in acc_normal]
    norm_train = [b for (a,b,c) in acc_normal]
    norm_test = [c for (a,b,c) in acc_normal]
    ave_train = [b for (a,b,c) in acc_ave]
    ave_test = [c for (a,b,c) in acc_ave]

    plt.figure()
    plt.plot(ep , norm_train, color="green", linestyle=":", label="Normal Accuracy (train)")
    plt.plot(ep, ave_train, color="red", linestyle=":", label="Average Accuracy (train)")

    plt.plot(ep , norm_test, color="green", label="Normal Accuracy (test)")
    plt.plot(ep, ave_test, color="red", label="Average Accuracy (test)")

    plt.legend()
    #plt.title ( "Accuracy on Training Set")
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("Training.pdf")
    plt.show()


    # plt.figure()
    # plt.plot(ep , norm_test, label="Normal Accuracy")
    # plt.plot(ep, ave_test, label="Average Accuracy")
    # plt.legend()
    # plt.title( "Accuracy on Test Set")
    # plt.xlabel("No. of Epochs")
    # plt.ylabel("Accuracy")
    # plt.savefig("Test.pdf")
    # plt.show()



    print("done")
