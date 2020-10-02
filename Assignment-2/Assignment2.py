
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# --Exercise 1 #----------------------------------------------------------------------------------------

#---------IMPORT DATA------------------------------------------------------------------------------------
test_data=np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
train_data=np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')

x_train=train_data[:,:-1]
y_train=train_data[:,-1]
x_test=test_data[:,:-1]
y_test=test_data[:,-1]

#-- Own Implementation -----------------------------------------------------------------------------------

# predict function: gives a label prediction for a single data point
def predict(X_train, y_train, x_test, k):
    # create list for distances and labels
    distances = []
    labels = []

    for i in range(len(X_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list into increasing order
    distances = sorted(distances)

    # make a list of the k neighbors' labels
    for i in range(k): # k is the number of neighbors we give as inputs
        index = distances[i][1] #nested list
        labels.append(y_train[index])

    # return the most common label
    return Counter(labels).most_common(1)[0][0]

# the function itself, it finds the best predicted labels for all data points
def knearneigh(X_train, y_train, X_test,k):
    predictions=[]
    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))
        
    return predictions

#---Using the built in function
classifier = KNeighborsClassifier(n_neighbors=1)  
classifier.fit(x_train, y_train) 
#make prediction on our dataset
y_pred=classifier.predict(x_test)

print(y_pred)

#-- Calculating testing accuracy--------------------------------------------------------------------
from sklearn.metrics import accuracy_score

predictions=knearneigh(x_train,y_train,x_test,1)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

#alternative way of calculating testing accuracy
cpr=y_test==y_pred
sum(cpr)
accuracy=sum(cpr)/cpr.shape
print('The testing accuracy of the model is', accuracy)

#--calculate training accuracy --------------------------------------------------------------------------
predictions=knearneigh(x_train,y_train,x_train,1)
accuracy = accuracy_score(y_train, predictions)
print('The training accuracy of the model is',accuracy)

#--EXERCISE 2-------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import KFold
# create indices for CV 
cv = KFold( n_splits =5) 
result = []

XTrain=x_train
YTrain=y_train

# loop over CV folds 
for train , test in cv.split (x_train):
    XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain [ train ],XTrain [ test ],YTrain [ train ],YTrain [ test ]
    
    for i in range(6):
        classifier = KNeighborsClassifier(n_neighbors=2*i+1)  
        classifier.fit(XTrainCV, YTrainCV) 

#make prediction on our dataset
        y_pred=classifier.predict(XTestCV)
        accTest=accuracy_score(y_pred,YTestCV) # calculating accuracy
        result.append(accTest) # collect the accuracy scores in a list


result=np.resize(result,(5,6)) #transform list into array
mean_accuracy=np.mean(result,axis=0) #mean accuracies coloumn wise for each hyperparameter
mean_accuracy
  
#Misclassification error
MSE = [1 - x for x in mean_accuracy] # getting the classification error 
a=list([1,3,5,7,9,11])
plt.plot(a,MSE) 
plt.xlabel("Hyperparameter") #give title to x axis
plt.ylabel("Classification error")


#--Exercise 3---------------------------------------------------------------------------------------------------------------

#built in
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(x_train, y_train) 

#make prediction on our testing dataset
predictions=knearneigh(x_train,y_train,x_test,3)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# training accuracy----------------------------------------------------------------------------

#my version
predictions=knearneigh(x_train,y_train,x_train,3)
accuracy = accuracy_score(y_train, predictions)
print('The training accuracy of the model with k=3 is',accuracy)

#built in function
y_pred2=classifier.predict(x_train)
accTest2=accuracy_score(y_pred2,y_train)
print('The training accuracy of the model with k=3 is',accTest2)

# testing accuracy ---------------------------------------------------------------------------------

#my version
predictions=knearneigh(x_train,y_train,x_test,3)
accuracy = accuracy_score(y_test, predictions)
print('The testing accuracy of the model with k=3 is',accuracy)

#built in function
y_pred=classifier.predict(x_test)
accTest=accuracy_score(y_pred,y_test)
print('The testing accuracy of the model with k=3 is',accTest)

#there is a big issue of testing on your training data

#--Exercise 4 -----------------------------------------------------------------------------------------------------------

# Normalization-------------------------------------------------------------------
from sklearn import preprocessing
from sklearn.model_selection import KFold

scaler=preprocessing.StandardScaler().fit(x_train)
XTrainN=scaler.transform(x_train)
XTestN = scaler.transform(x_test)

# Cross-Validation for best Hyperparameter ---------------------------------------
cv = KFold( n_splits =5) 
result = []

XTrain=XTrainN
YTrain=y_train

for train , test in cv.split (x_train):
    XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain [ train ],XTrain [ test ],YTrain [ train ],YTrain [ test ]
    
    for i in range(6):
        classifier = KNeighborsClassifier(n_neighbors=2*i+1)  
        classifier.fit(XTrainCV, YTrainCV) 

#make prediction on our dataset
        y_pred=classifier.predict(XTestCV)
        accTest=accuracy_score(y_pred,YTestCV) # calculating accuracy
        result.append(accTest) # collect the accuracy scores in a list
        
result=np.resize(result,(5,6))
mean_accuracy=np.mean(result,axis=0) #mean accuracies coloumn wise
mean_accuracy #best hyperparameter is k=3

#training accuracy with Normalization ------------------------------------------------------------------------
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(XTrainN, y_train) 
y_pred=classifier.predict(XTrainN)
accTest=accuracy_score(y_pred,y_train)
print('The training accuracy of the model with normalization is',accTest)

#test accuracy with Normalization---------------------------------------------------------------------------------
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(XTrainN, y_train) 
y_pred=classifier.predict(XTestN)
accTest=accuracy_score(y_pred,y_test)
print('The testing accuracy of the model with normalization is',accTest)
#------------------------------------------------------------------------------------