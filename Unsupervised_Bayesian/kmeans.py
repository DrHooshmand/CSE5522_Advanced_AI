#!/usr/bin/python

#
# CSE 5522: HW 2 K-Means Implementation
# Author: Eric Fosler-Lussier
# 

from numpy import *
from matplotlib.pyplot import *
import json
from numpy.random import *
from prettytable import PrettyTable
from tabulate import tabulate

if sys.version_info[0] == 2:
    range = range

colors=['b+','r+','g+','m+','k+','b*','r*','g*','m*','k*','bo','ro','go','mo','ko'];
epsilon=0.0001
block_with_plot=True

def kmeans_iter(curmeans,data):
    meanassign=kmeans_pointassignments(curmeans,data)
    newmeans=curmeans.copy()
    for i in range(0,len(curmeans)):
        assignedpoints=compress(meanassign==i,data,axis=0)
        if len(assignedpoints)==0:
            newmeans[i]=randn(1,data.shape[1])*std(data,axis=0)+mean(data,axis=0)
        else:
            newmeans[i]=average(assignedpoints,axis=0)
    return newmeans

def kmeans_pointassignments(curmeans,data):
    dist=zeros([len(data),len(curmeans)])
    for i in range(0,len(curmeans)):
        dist[:,i]=linalg.norm(data-curmeans[i],axis=1)
    meanassign=argmin(dist,axis=1)
    return meanassign

# runs k-means for the given number of means and data points
def kmeans(nmeans,data):
    means=randn(nmeans,data.shape[1])*std(data,axis=0)+mean(data,axis=0)
    while True:
        newmeans=kmeans_iter(means,data)
        dist=sum(linalg.norm(means-newmeans,axis=0))
        means=newmeans
        if dist<epsilon:
            break
    return newmeans

def plot_assignments(curmeans,data,labels):
    clf()
    curassign=kmeans_pointassignments(curmeans,data)
    for i in range(0,curmeans.shape[0]):
        tp=compress(curassign==i,data,axis=0)
        plot(tp[:,0],tp[:,1],colors[i])
    for ((x,y),lab) in zip(data,labels):
        text(x+.03, y+.03, lab, fontsize=9)
    plot(curmeans[:,0],curmeans[:,1],'c^',markersize=12)
    show(block=block_with_plot)


def tabular(means, data, labels):
    curassign = kmeans_pointassignments(means, data)
    n_vect = len(means)
    n_labels = max(labels) + 1
    tab = np.zeros((n_vect, n_labels))

    for i in range(len(labels)):
        j = curassign[i]
        c = traininglabels[i]
        tab[j, c] += 1

    C = [0] * n_labels
    for i in range(len(C)):
        # C[i] = list(traininglabels).count(i) / len(traininglabels)
        C[i] = list(labels).count(i)
    C = repeat(array([C]), n_vect, axis=0)
    prob_tab = divide(tab, C)
    return prob_tab

def Pc (labels):
    n_labels = max(labels) + 1
    C = [0] * n_labels
    for i in range(len(C)):
        C[i]=list(labels).count(i)/ len(labels)
    return C

def accuracy(algo,given):
    compare = []
    for i in range(len(algo)):
        if algo[i] == given[i]:
            compare.append(1)
        else:
            compare.append(0)
    return sum(compare) / len(compare) * 100

def test (means, data_test, labels_test, trained_probs):
    P_c = Pc(labels_test)
    test_assign = kmeans_pointassignments(means, data_test)
    kmean_test_label = []
    for i in range(len(data_test)):
        C_dum = []
        for j in range(max(labels_test)+1):
            C_dum.append(trained_probs[test_assign[i], j] * P_c[j])
        kmean_test_label.append(C_dum.index(amax(C_dum)))
    return kmean_test_label


#seed(1)
f=open('training.json','r')
trainingdata=json.load(f)
traininglabels=array(trainingdata['labels'])
trainingpoints=array(trainingdata['points'])
f.close()

f=open('testing.json','r')
testingdata=json.load(f)
testinglabels=array(testingdata['labels'])
testingpoints=array(testingdata['points'])
f.close()

print('running k-means')

#print('plotting assignments of training points')
#plot_assignments(vectormeans,trainingpoints,traininglabels)

vectormeans=kmeans(10,trainingpoints)
prob_tab = tabular(vectormeans,trainingpoints,traininglabels)


# for i in range(shape(prob_tab)[0]):
#     for j in range(shape(prob_tab)[1]):
#         print(str(prob_tab[i,j]) + ",")

x = PrettyTable()
for row in range(0, shape(prob_tab)[0]):
    init = "V:"+str(row+1)
    a = list(prob_tab[row,:])
    a.insert(0,init)
    x.add_row(a)
x.field_names = ["V/C","C0","C1","C2","C3","C4","C5","C6"]
print(x)

P_c = Pc(traininglabels)
test_assign = kmeans_pointassignments(vectormeans,testingpoints)
kmean_test_label = []
for i in range( len(testingpoints) ):
    C_dum =[]
    for j in range(max(testinglabels)):
        C_dum.append(prob_tab[test_assign[i] , j] * P_c[j] )
    kmean_test_label.append( C_dum.index(amax (C_dum) ) )

kmean_test_label = test(vectormeans,testingpoints,testinglabels,prob_tab)
acc = accuracy(kmean_test_label,testinglabels)

#### Step 3
print("Step 3 Begins:")
acc_error= []
for i in range(10):
    vectormeans = kmeans(10, trainingpoints)
    prob_tab = tabular(vectormeans, trainingpoints, traininglabels)
    kmean_test_label = test(vectormeans, testingpoints, testinglabels, prob_tab)
    acc = accuracy(kmean_test_label, testinglabels)
    acc_error.append(acc)
print("Accuracy results over 10 iterations: %s" %acc_error )
print("Mean = %s pct and STD= %s"  % ( mean(acc_error) , std(acc_error) ) )

#### Step 4
print("Step 4 Begins:")
K = [2,5,6,8,12,15,20,50]
stat = []
for k in K:
    acc_error = []
    for i in range(10):
        vectormeans = kmeans(k, trainingpoints)
        prob_tab = tabular(vectormeans, trainingpoints, traininglabels)
        kmean_test_label = test(vectormeans, testingpoints, testinglabels, prob_tab)
        acc = accuracy(kmean_test_label, testinglabels)
        acc_error.append( (acc) )
    stat.append( (mean(acc_error), std(acc_error)) )
    print("K= %s Mean = %s pct and STD= %s" % (k, mean(acc_error) , std(acc_error)))



#### Step 5
print("Step 5 Begins:")

Gauss = [3.2,4,6,8,10]

stat= []
for gauss in Gauss:
    name_train = 'hw2_training_'+str(gauss)+'.json'
    name_test = 'hw2_testing_'+str(gauss)+'.json'

    f=open(name_train,'r')
    trainingdata=json.load(f)
    traininglabels=array(trainingdata['labels'])
    trainingpoints=array(trainingdata['points'])
    f.close()

    f=open(name_test,'r')
    testingdata=json.load(f)
    testinglabels=array(testingdata['labels'])
    testingpoints=array(testingdata['points'])
    f.close()

    acc_error = []
    for i in range(10):
        vectormeans = kmeans(10, trainingpoints)
        prob_tab = tabular(vectormeans, trainingpoints, traininglabels)
        kmean_test_label = test(vectormeans, testingpoints, testinglabels, prob_tab)
        acc = accuracy(kmean_test_label, testinglabels)
        acc_error.append( (acc) )
    stat.append( (gauss, mean(acc_error), std(acc_error)) )
    print("Gauss = %s Mean = %s pct and STD= %s" % (gauss, mean(acc_error) , std(acc_error)))

#### Bonus 1
print("Bonus 1 Begins:")
name_train = 'hw2_training_'+"Bon1"+'.json'
name_test = 'hw2_testing_'+"Bon1"+'.json'

f=open(name_train,'r')
trainingdata=json.load(f)
traininglabels=array(trainingdata['labels'])
trainingpoints=array(trainingdata['points'])
f.close()

f=open(name_test,'r')
testingdata=json.load(f)
testinglabels=array(testingdata['labels'])
testingpoints=array(testingdata['points'])
f.close()
acc_error= []
for i in range(10):
    vectormeans = kmeans(10, trainingpoints)
    prob_tab = tabular(vectormeans, trainingpoints, traininglabels)
    kmean_test_label = test(vectormeans, testingpoints, testinglabels, prob_tab)
    acc = accuracy(kmean_test_label, testinglabels)
    acc_error.append(acc)
print("Mean = %s pct and STD= %s"  % ( mean(acc_error) , std(acc_error) ) )


#### Bonus 2
print("Bonus 2 Begins:")

Dims = [2,3,4,5]

stat= []
for dims in Dims:
    name_train = 'hw2_training_'+str(dims)+"dim"+'.json'
    name_test = 'hw2_testing_'+str(dims)+"dim"+'.json'

    f=open(name_train,'r')
    trainingdata=json.load(f)
    traininglabels=array(trainingdata['labels'])
    trainingpoints=array(trainingdata['points'])
    f.close()

    f=open(name_test,'r')
    testingdata=json.load(f)
    testinglabels=array(testingdata['labels'])
    testingpoints=array(testingdata['points'])
    f.close()

    acc_error = []
    for i in range(10):
        vectormeans = kmeans(10, trainingpoints)
        prob_tab = tabular(vectormeans, trainingpoints, traininglabels)
        kmean_test_label = test(vectormeans, testingpoints, testinglabels, prob_tab)
        acc = accuracy(kmean_test_label, testinglabels)
        acc_error.append( (acc) )
    stat.append( (dims, mean(acc_error), std(acc_error)) )
    print("Dim = %s Mean = %s pct and STD= %s" % (dims, mean(acc_error) , std(acc_error)))


# header = []
# for i in range(shape(prob_tab)[0]):
#     header.append("V"+str(i+1))
# table = tabulate(prob_tab)

print("done")
