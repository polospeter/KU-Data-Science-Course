
# Exam - Final assignment

import numpy as np
import matplotlib.pyplot as plt

# Exercise 2 - Linear Regression #######################################################################################################

datatrain=np.loadtxt('redwine_training.txt')
datatest=np.loadtxt('redwine_testing.txt')

#--My implementation:  -----------------------------------------------------------------------------------------------------------------
def multivarlinreg(X,y):
    dep=y # dependent variables
    indep=X # independent variables
# add a column of ones to the indep
    num_pts = indep.shape[0] # rows, columns  # fix here incase it is only one column
    onevec = np.ones((num_pts,1))
    indep = np.concatenate((onevec, indep), axis = 1) 

    x=(indep.T).dot(indep)
    inverse = np.linalg.inv(x)
    # xx=np.dot(inverse,indep.T)
    xx=np.dot(dep.T,indep) 
    w=np.dot(inverse,xx) #weights

    prediction=np.dot(w,indep.T)
    
    return w, prediction

weights,predicts=multivarlinreg(datatrain[:,0:11],datatrain[:,11])

# makes the exact same predictions as the built in Regression function
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(datatrain[:,0:11],datatrain[:,11])
predictionss = lm.predict(datatrain[:,0:11])
print(lm.coef_) # same as my weights
print(lm.intercept_) # w0- indeed have the same results
print(weights)

# part b) ===============================================================================================================
indep=datatrain[:,0].reshape((1000,1)) # had to reshape my data first
weights,predicts=multivarlinreg(indep,datatrain[:,11]) # only first feature
print(weights)

lm.fit(indep,datatrain[:,11])
print(lm.coef_) # same as my weights
print(lm.intercept_)

# part c) ================================================================================================================
weights,predicts=multivarlinreg(datatrain[:,0:11],datatrain[:,11])
print(weights) # the weights

# Exercise-3 ##############################################################################################################x
# Root mean square error

def rmse(f, t):
    a=f-t
    rmse=np.sqrt(np.mean(np.square(np.absolute(a))))
    return rmse

# part b) ============================================================================================================
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

dep=datatrain[:,11] #dependent variable
indep=datatrain[:,0] # independent variable

# reshape them
dep=dep.reshape((1000,1))
indep=indep.reshape((1000,1))
       
lm.fit(indep,dep)
predictionss = lm.predict(indep)
print(lm.coef_)

rmse(predictionss,dep)

# Compare results with built-in function
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(dep, predictionss))
print(rms)
print(rmse(predictionss,dep)) # they produce the same results

# part c) ===========================================================================================================

dep=datatrain[:,11]
indep=datatrain[:,0:11]

dep=dep.reshape((1000,1))

lm.fit(indep,dep)
predictions = lm.predict(indep)

rmse(predictions,dep) #smaller error, not surprising since with more independent variables we always getting a more accurate prediction

# Exercise-5 ############################################################################################################
from sklearn.ensemble import RandomForestClassifier

# import data ----------------------------------------------
test_data=np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
train_data=np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')

x_train=train_data[:,:-1]
y_train=train_data[:,-1]
x_test=test_data[:,:-1]
y_test=test_data[:,-1]
#----------------------------------------------------------

clf = RandomForestClassifier(n_estimators=50) # with 50 trees
clf.fit(x_train, y_train)

# apply prediction to data it has not seen before
pred=clf.predict(x_test) #predictions

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)
print(accuracy)  # 0.968 quite good, has to compare with result from previous assignment

# Does it beat the nearest neighbor classi
er?
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(x_train, y_train) 
#make prediction on our dataset
y_pred=classifier.predict(x_test)

print(y_pred)

#-- Calculating testing accuracy-------
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy) #0.9494 so the random forest is doing slightly better

# Exercise 6 - Gradient descent ######################################################################################

#------------------------------------------------------------------------------------------
next_x = 1  # We start the search at x=6
rate = 0.01  # Step size multiplier- This is the LEARNING RATE
precision = 10**-10  # Desired precision of result
max_iters = 10000  # Maximum number of iterations

def gradient(next_x,rate,precision,max_iters):
    
# Derivative of the function -------------------------------------------------------
    df=lambda x: np.exp(-x/2)*(-0.5)+20*x

    distances = [] # create array
    steps=[]
    for i in range(max_iters):
        steps.append([next_x]) #collect the steps
        
        current_x = next_x
        next_x = current_x - rate * df(current_x)
        step = next_x - current_x
    
        if abs(step) <= precision:
            break
        
    return next_x,len(steps),steps  # the number of steps it took to converge to the minimum
#---------------------------------------------------------------------------------- 

print("Minimum at", next_x) #0.02469323262707432

# The output for the above will be something like
print(gradient(1,0.1,10**-10,10000)) # I guess this converges to infinity thats why it will be an error
print(gradient(1,0.01,10**-10,10000))
print(gradient(1,0.001,10**-10,10000))
print(gradient(1,0.0001,10**-10,10000))

aa=gradient(1,0.01,10**-10,10000)
stepps=aa[2]

# derivative
df=lambda x: np.exp(-x/2)*(-0.5)+20*x

# original function
f=lambda x: np.exp(-x/2)+10*x**2

# part b) ------- TANGENT LINES FOR THE 4 LEARNING RATES -------------------------------------------------------------------------------------------------------

stepps=gradient(1,0.1,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.1) # x-axis
plt.plot(x,f(x))
plt.plot(x,f(myarray[0])+df(myarray[0])*(x-myarray[0])) # nice fit
plt.plot(x,f(myarray[1])+df(myarray[1])*(x-myarray[1])) # nice fit
plt.plot(x,f(myarray[2])+df(myarray[2])*(x-myarray[2]))
plt.plot(x,f(myarray[3])+df(myarray[3])*(x-myarray[3]))
plt.scatter(myarray[0:4],f(myarray[0:4]))
plt.title("Tangent lines with learning rate 0.1")

#--------------------------------------------------------------------------------------------------------
stepps=gradient(1,0.01,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.01) # x-axis
plt.plot(x,f(x))
plt.plot(x,f(myarray[0])+df(myarray[0])*(x-myarray[0])) # nice fit
plt.plot(x,f(myarray[1])+df(myarray[1])*(x-myarray[1])) # nice fit
plt.plot(x,f(myarray[2])+df(myarray[2])*(x-myarray[2]))
plt.plot(x,f(myarray[3])+df(myarray[3])*(x-myarray[3]))
plt.scatter(myarray[0:4],f(myarray[0:4]))
plt.title("Tangent lines with learning rate 0.01")

#--------------------------------------------------------------------------------------------------------
stepps=gradient(1,0.001,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.001) # x-axis
plt.plot(x,f(x))
plt.plot(x,f(myarray[0])+df(myarray[0])*(x-myarray[0])) # nice fit
plt.plot(x,f(myarray[1])+df(myarray[1])*(x-myarray[1])) # nice fit
plt.plot(x,f(myarray[2])+df(myarray[2])*(x-myarray[2]))
plt.plot(x,f(myarray[3])+df(myarray[3])*(x-myarray[3]))
plt.scatter(myarray[0:4],f(myarray[0:4]))
plt.title("Tangent lines with learning rate 0.001")

#--------------------------------------------------------------------------------------------------------
stepps=gradient(1,0.0001,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.0001) # x-axis
plt.plot(x,f(x))
plt.plot(x,f(myarray[0])+df(myarray[0])*(x-myarray[0])) # nice fit
plt.plot(x,f(myarray[1])+df(myarray[1])*(x-myarray[1])) # nice fit
plt.plot(x,f(myarray[2])+df(myarray[2])*(x-myarray[2]))
plt.plot(x,f(myarray[3])+df(myarray[3])*(x-myarray[3]))
plt.scatter(myarray[0:4],f(myarray[0:4]))
plt.title("Tangent lines with learning rate 0.0001")

# part c)  visualize the first 10 steps ===========================================================================================================

stepps=gradient(1,0.1,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.1) # x-axis
plt.plot(x,f(x))
plt.scatter(myarray[0:10],f(myarray[0:10]),color="Red") # nice fit
plt.title("First 10 gradient descent steps with learning rate 0.1")
# ----------------------------------------------

stepps=gradient(1,0.01,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.1) # x-axis
plt.plot(x,f(x))
plt.scatter(myarray[0:10],f(myarray[0:10]),color="Red") # nice fit
plt.title("First 10 gradient descent steps with learning rate 0.01")
# ----------------------------------------------

stepps=gradient(1,0.001,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.1) # x-axis
plt.plot(x,f(x))
plt.scatter(myarray[0:10],f(myarray[0:10]),color="Red") 
plt.title("First 10 gradient descent steps with learning rate 0.001")

# ----------------------------------------------

stepps=gradient(1,0.0001,10**-10,10000)[2]

myarray = np.asarray(stepps) # convert list to array

x=np.arange(-2,2,0.1) # x-axis
plt.plot(x,f(x))
plt.scatter(myarray[0:10],f(myarray[0:10]),color="Red") # nice fit
plt.title("First 10 gradient descent steps with learning rate 0.0001")

# part d) ------------------------------------------------------------------------------------------------------------------

print("In case of learning rate 0.1 the minimum value is ",gradient(1,0.1,10**-10,10000)[0]," after ",gradient(1,0.1,10**-10,10000)[1]," iterations") # I guess this converges to infinity thats why it will be an error
print("In case of learning rate 0.01 the minimum value is",gradient(1,0.01,10**-10,10000)[0],"after",gradient(1,0.01,10**-10,10000)[1]," iterations")
print("In case of learning rate 0.001 the minimum value is",gradient(1,0.001,10**-10,10000)[0],"after",gradient(1,0.001,10**-10,10000)[1]," iterations")
print("In case of learning rate 0.0001 the minimum value is",gradient(1,0.0001,10**-10,10000)[0],"after",gradient(1,0.0001,10**-10,10000)[1]," iterations")

# EXERCISE 7 --lOGISTIC REGRESSION ####################################################################################

# Part a) ===========================================================================================================

# load data- Iris dataset
datatrain1=np.loadtxt('Iris2D1_train.txt')
datatest1=np.loadtxt('Iris2D1_test.txt')

datatrain2=np.loadtxt('Iris2D2_train.txt')
datatest2=np.loadtxt('Iris2D2_test.txt')

#--------------------------------------------------------------------------------
label0=(datatrain1[:,2]==0)
label1=(datatrain1[:,2]==1)

data0=datatrain1[label0] #masks
data1=datatrain1[label1]

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.scatter(data0[:,0],data0[:,1],color='orange') #class 0
plt.scatter(data1[:,0],data1[:,1],color='blue') #class 1
plt.title("Plot of Iris2D1_train dataset")
oran_patch = mpatches.Patch(color='orange', label='Class 0')
blue_patch = mpatches.Patch(color='blue', label='Class 1')
plt.legend(handles=[blue_patch,oran_patch])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

#---------------------------------------------------------------------------------
label0=(datatrain2[:,2]==0)
label1=(datatrain2[:,2]==1)

data0=datatrain2[label0]
data1=datatrain2[label1]

plt.scatter(data0[:,0],data0[:,1],color='orange') #class 0
plt.scatter(data1[:,0],data1[:,1],color='blue') #class 1
plt.title("Plot of Iris2D2_train dataset")
oran_patch = mpatches.Patch(color='orange', label='Class 0')
blue_patch = mpatches.Patch(color='blue', label='Class 1')
plt.legend(handles=[blue_patch,oran_patch])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
#---------------------------------------------------------------------------------

label0=(datatest1[:,2]==0)
label1=(datatest1[:,2]==1)

data0=datatest1[label0]
data1=datatest1[label1]

plt.scatter(data0[:,0],data0[:,1],color='orange') #class 0
plt.scatter(data1[:,0],data1[:,1],color='blue') #class 1
plt.title("Plot of Iris2D1_test dataset")
oran_patch = mpatches.Patch(color='orange', label='Class 0')
blue_patch = mpatches.Patch(color='blue', label='Class 1')
plt.legend(handles=[blue_patch,oran_patch])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

#----------------------------------------------------------------------------

label0=(datatest2[:,2]==0)
label1=(datatest2[:,2]==1)

data0=datatest2[label0]
data1=datatest2[label1]

plt.scatter(data0[:,0],data0[:,1],color='orange') #class 0
plt.scatter(data1[:,0],data1[:,1],color='blue') #class 1
plt.title("Plot of Iris2D2_test dataset")
oran_patch = mpatches.Patch(color='orange', label='Class 0')
blue_patch = mpatches.Patch(color='blue', label='Class 1')
plt.legend(handles=[blue_patch,oran_patch])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Part b) ===========================================================================================================

# Sigmoid function
#theta=lambda x: np.exp(x)/(1+np.exp(x))
theta=lambda x: 1.0 / (1 + np.exp(-x))

# log likelihood function----------------------------------
def logistic_insample(X, y, w):
    N, num_feat = X.shape    
    E = 0
    for n in range(N):
        E = E + np.log(1+np.exp(-y[n]*(X[n].dot(w))))
    E=E/N
    return E
#-------------------------------------------------------------

def logistic_gradient(X, y, w):
    N, _ = X.shape
   # g = np.mean(-y*X.T@theta(-y*np.dot(X,w)))
    g = 0*w
    
    for n in range(N):
        g = g+y[n]*X[n]*theta(-y[n]*np.dot(X[n],w))
        
    g=-g/N
    return g
#--------------------------------------------------------------------------------

def log_reg(Xorig, y, max_iter, grad_thr):   
    
    num_pts, num_feat = Xorig.shape # rows, columns
    onevec = np.ones((num_pts,1))
    X = np.concatenate((onevec, Xorig), axis = 1) # adding a column of ones
    dplus1 = num_feat + 1
        
    # y is a N by 1 matrix of target values -1 and 1
    y = np.array((y-.5)*2)
        
    # Initialize learning rate for gradient descent
    learningrate = 0.00      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
    
    # Starting values for w   
    w = 0.1*np.random.randn(num_feat + 1) 
    
    # Compute value of logistic log likelihood
    value = logistic_insample(X,y,w)
    
    num_iter = 0  
    convergence = 0
    
    # Keep track of function values
    E_in = []
    w_new=w
    
    while convergence == 0:
        num_iter = num_iter + 1                        
        # Compute gradient at current w      
        g = logistic_gradient(X,y,w)
        # Set direction to move and take a step       
        v=-g
        
        w_new = w_new+v*learningrate
        
        # Compute in-sample error for new w
        cur_value = logistic_insample(X,y,w_new)
        
        if cur_value < value:
            w = w_new
            value = cur_value
            E_in.append(value)
            learningrate *=1.1
        else:
            learningrate *= 0.9   
            
        # Determine whether we have converged: Is gradient norm below
        # threshold, and have we reached max_iter?
        g_norm = np.linalg.norm(g)
        if g_norm < grad_thr: # check threshold
            convergence = 1
        elif num_iter > max_iter: # if the iterations surpassed the limit
            convergence = 1
           
    return w, E_in 

#-------------------------------------------------------------------------------------------------

w, E = log_reg(datatrain1[:,0:2],datatrain1[:,2], 20000, 0.0000)

Xorig=datatest1[:,0:2] 
Xorig=datatrain1[:,0:2]

# taking input the training set data matrix, the training set label vector, and the test set data matrix
def log_prediction(Xorig,y,Xtest):
    max_iter=20000 # set default values
    grad_thr=0.0000
    w, E = log_reg(Xorig,y,max_iter, grad_thr) # getting the weights
    N, d = Xtest.shape
    N1 = np.reshape(np.ones(N), (N, 1))
    X = np.hstack((N1, Xtest))
    # the probabilities:
    result=theta(np.dot(X,w)) # i should be getting values between 0 and 1
    label0=(result>0.5)
    label0 = label0*1 # convert to numeric
    return label0

from sklearn.metrics import accuracy_score

# Dataset 1

# testing error ---------------------------------------
label0=log_prediction(datatrain1[:,0:2],datatrain1[:,2],datatest1[:,0:2]) # the predicted labels
accuracy = accuracy_score(datatest1[:,2],label0)
print(accuracy)
print("The testing error is",1-accuracy)
w, E = log_reg(datatrain1[:,0:2],datatrain1[:,2], 20000, 0.0000)
print(w) #weights

#training error ---------------------------------------
label0=log_prediction(datatrain1[:,0:2],datatrain1[:,2],datatrain1[:,0:2])
accuracy1 = accuracy_score(datatrain1[:,2],label0)
print(accuracy1)
print("The training error is",1-accuracy)

#----------------------------------------------------------------------------------------------------------------
# Dataset 2

#testing accuracy
labels=log_prediction(datatrain2[:,0:2],datatrain2[:,2],datatest2[:,0:2]) # the predicted labels
accuracy = accuracy_score(datatest2[:,2],labels)
print(1-accuracy)
w, E = log_reg(datatrain2[:,0:2],datatrain2[:,2], 20000, 0.0000)
print(w) #weights

#training accuracy
labels=log_prediction(datatrain2[:,0:2],datatrain2[:,2],datatrain2[:,0:2]) # the predicted labels
accuracy = accuracy_score(datatrain2[:,2],labels)
print(1-accuracy)


#------------------------------------------------------------------------------------------------------------------
# Testing with built-in log reg function ---------------------------------------------------------
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(datatrain1[:,0:2],datatrain1[:,2])
clf.coef_
res=clf.predict(datatest1[:,0:2])
accuracy = accuracy_score(datatest1[:,2],res)
print(accuracy)


# ====================================================================================================================

# EXERCISE 9 ##########################################################################################################
digits=np.loadtxt('MNIST_179_digits.txt')
labels=np.loadtxt('MNIST_179_labels.txt')

from sklearn.cluster import KMeans 
from PIL import Image # for visualizing the cluster centers

# Comparison - they indeed produce the same results
kmeans=KMeans(n_clusters=3).fit(digits)
centers=kmeans.cluster_centers_ # so we got 3 cluster centers
kmeans.labels_
print(kmeans.cluster_centers_)

# separate data into cluster labels
cluster0=(kmeans.labels_==0)
cluster1=(kmeans.labels_==1)
cluster2=(kmeans.labels_==2)

group0=labels[cluster0]
group1=labels[cluster1]
group2=labels[cluster2]

# ratio of number in the 3 groups:

# center 0 -----------------this is the group 7
group0digit1=(group0==1)
group0digit1=sum(group0digit1)/len(group0)
print("Ratio of 1s in cluster is",group0digit1)

group0digit7=(group0==7)
group0digit7=sum(group0digit7)/len(group0)
print("Ratio of 7s in cluster is",group0digit7)

group0digit9=(group0==9)
group0digit9=sum(group0digit9)/len(group0)
print("Ratio of 9s in cluster is",group0digit9)

# Visualize cluster center---------------------------
num=kmeans.cluster_centers_[0].reshape(28,28)
img = Image.fromarray(num)

# enlarge the image a bit
size = 300, 300
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()
#_______________________________________________________________________________________
# center 1 ----------- there are the 1s in this cluster
group1digit1=(group1==1)
group1digit1=sum(group1digit1)/len(group1)
print("Ratio of 1s in cluster is",group1digit1)

group1digit7=(group1==7)
group1digit7=sum(group1digit7)/len(group1)
print("Ratio of 7s in cluster is",group1digit7)

group1digit9=(group1==9)
group1digit9=sum(group1digit9)/len(group1)
print("Ratio of 9s in cluster is",group1digit9)

#----------------------------------------------------
num=kmeans.cluster_centers_[1].reshape(28,28)
img = Image.fromarray(num)

# enlarge the image a bit
size = 300, 300
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()
#_________________________________________________________________________________________
# center 2 --------------- this the group of 9s
group2digit1=(group2==1)
group2digit1=sum(group2digit1)/len(group2)
print("Ratio of 1s in cluster is",group2digit1)

group2digit7=(group2==7)
group2digit7=sum(group2digit7)/len(group2)
print("Ratio of 7s in cluster is",group2digit7)

group2digit9=(group2==9)
group2digit9=sum(group2digit9)/len(group2)
print("Ratio of 9s in cluster is",group2digit9)

#----------------------------------------------------
num=kmeans.cluster_centers_[2].reshape(28,28)
img = Image.fromarray(num)

# enlarge the image a bit
size = 300, 300
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

# part b) ==============================================================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cross-Validation for best Hyperparameter -------------------------------------------
from sklearn.model_selection import KFold

digits=np.loadtxt('MNIST_179_digits.txt')
labels=np.loadtxt('MNIST_179_labels.txt')

def crossvalpar(XTrain,YTrain):
    
    cv = KFold( n_splits =5) 
    result = []

    for train , test in cv.split (XTrain):
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
    return mean_accuracy 

print(crossvalpar(digits,labels)) # k=1 seem to have the best accuracy

#plot here:

#========================================================================================================================

# Exercise 10 ###########################################################################################################

# part a) ============================================================================================================
from sklearn.decomposition import PCA 

dataset=digits
pca = PCA()
pca.fit(dataset)
eig=pca.explained_variance_ #eigenvalues are the same 
print(eig)

plt.plot(np.cumsum(eig)/np.sum(eig)) # somewhere between 200-300 PCs the curve flattens out
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance in %")
plt.title("Cumulative explained variance of PCs")

# part b) =============================================================================================================

# 3 CLUSTER centers as images

from PIL import Image

# PCA with 20 components
dataset=digits
pca = PCA(n_components=20) 
pca.fit(dataset)
pc=pca.components_ 
projected=dataset.dot(pc.T) # project data to PCA components

# do K-means clustering -------------------------------------------------------------------------------------
kmeans=KMeans(n_clusters=3).fit(projected)
centers=kmeans.cluster_centers_ # so we got 3 cluster centers
kmeans.labels_
print(kmeans.cluster_centers_)

# separate data into cluster labels
cluster0=(kmeans.labels_==0)
cluster1=(kmeans.labels_==1)
cluster2=(kmeans.labels_==2)

group0=labels[cluster0]
group1=labels[cluster1]
group2=labels[cluster2]

# ratio of the digits in the 3 groups:

# center 0 -----------------this is the group of 7s
group0digit1=(group0==1)
group0digit1=sum(group0digit1)/len(group0)
print("Ratio of 1s in cluster is",group0digit1)

group0digit7=(group0==7)
group0digit7=sum(group0digit7)/len(group0)
print("Ratio of 7s in cluster is",group0digit7)

group0digit9=(group0==9)
group0digit9=sum(group0digit9)/len(group0)
print("Ratio of 9s in cluster is",group0digit9)
#_______________________________________________________________________________________
# center 1 ----------- there are the 1s in this cluster
group1digit1=(group1==1)
group1digit1=sum(group1digit1)/len(group1)
print("Ratio of 1s in cluster is",group1digit1)

group1digit7=(group1==7)
group1digit7=sum(group1digit7)/len(group1)
print("Ratio of 7s in cluster is",group1digit7)

group1digit9=(group1==9)
group1digit9=sum(group1digit9)/len(group1)
print("Ratio of 9s in cluster is",group1digit9)

#_________________________________________________________________________________________
# center 2 --------------- this the group of 9s
group2digit1=(group2==1)
group2digit1=sum(group2digit1)/len(group2)
print("Ratio of 1s in cluster is",group2digit1)

group2digit7=(group2==7)
group2digit7=sum(group2digit7)/len(group2)
print("Ratio of 7s in cluster is",group2digit7)

group2digit9=(group2==9)
group2digit9=sum(group2digit9)/len(group2)
print("Ratio of 9s in cluster is",group2digit9)
#----------------------------------------------------

# VISUALIZE CLUSTER CENTERS

# transform back the center to the original dimension
pcc=pca.components_ #eigenvectors

projcent = pca.inverse_transform(centers) #cluster centers in originial dimension

# Cluster center 1 ----------------------------------
num=projcent[0].reshape(28,28)
img = Image.fromarray(num)
# enlarge the image a bit
size = 300, 300
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

#--------------------------------------------------
# Cluster center 2
size = 300, 300
num=projcent[1].reshape(28,28)
img = Image.fromarray(num)
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

#--------------------------------------------------
# Cluster center 3
size = 300, 300
num=projcent[2].reshape(28,28)
img = Image.fromarray(num)
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

# PCA with 200 components ==========================================================================================

dataset=digits
pca = PCA(n_components=200) 
pca.fit(dataset)
pc=pca.components_ # project data to PCA components
projected=dataset.dot(pc.T)

# do K-means clustering -------------------------------------------------------------------------------------
kmeans=KMeans(n_clusters=3).fit(projected)
centers=kmeans.cluster_centers_ # so we got 3 cluster centers
kmeans.labels_
print(kmeans.cluster_centers_)

#---------------------------------------------------------------------------------------------------------------------

# separate data into cluster labels
cluster0=(kmeans.labels_==0)
cluster1=(kmeans.labels_==1)
cluster2=(kmeans.labels_==2)

group0=labels[cluster0]
group1=labels[cluster1]
group2=labels[cluster2]

# ratio of number int he 3 groups:

# center 0 -----------------this is the group of 9s
group0digit1=(group0==1)
group0digit1=sum(group0digit1)/len(group0)
print("Ratio of 1s in cluster is",group0digit1)

group0digit7=(group0==7)
group0digit7=sum(group0digit7)/len(group0)
print("Ratio of 7s in cluster is",group0digit7)

group0digit9=(group0==9)
group0digit9=sum(group0digit9)/len(group0)
print("Ratio of 9s in cluster is",group0digit9)
#_______________________________________________________________________________________
# center 1 ----------- there are the 1s in this cluster
group1digit1=(group1==1)
group1digit1=sum(group1digit1)/len(group1)
print("Ratio of 1s in cluster is",group1digit1)

group1digit7=(group1==7)
group1digit7=sum(group1digit7)/len(group1)
print("Ratio of 7s in cluster is",group1digit7)

group1digit9=(group1==9)
group1digit9=sum(group1digit9)/len(group1)
print("Ratio of 9s in cluster is",group1digit9)

#_________________________________________________________________________________________
# center 2 --------------- this the group of 9s
group2digit1=(group2==1)
group2digit1=sum(group2digit1)/len(group2)
print("Ratio of 1s in cluster is",group2digit1)

group2digit7=(group2==7)
group2digit7=sum(group2digit7)/len(group2)
print("Ratio of 7s in cluster is",group2digit7)

group2digit9=(group2==9)
group2digit9=sum(group2digit9)/len(group2)
print("Ratio of 9s in cluster is",group2digit9)

#----------------------------------------------------------------------------------------------------------------
# VISUALIZE CLUSTER CENTERS

# transform back the center to the original dimension
pcc=pca.components_ #eigenvectors

projcent = pca.inverse_transform(centers) #cluster centers in originial dimension

#----------------------------------------------------------
# Cluster center 0
num=projcent[0].reshape(28,28)
img = Image.fromarray(num)
# enlarge the image a bit
size = 300, 300
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

#---------------------------------------------------------
# Cluster center 1
size = 300, 300
num=projcent[1].reshape(28,28)
img = Image.fromarray(num)
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

#---------------------------------------------------------
# Cluster center 2
size = 300, 300
num=projcent[2].reshape(28,28)
img = Image.fromarray(num)
im = img
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.show()

#---------------------------------------------------------------------------------------------------------------------------
# part c)===============================================================================================================

# Cross-Validation for best Hyperparameter --------------------------------------
from sklearn.model_selection import KFold

def crossvalpar(XTrain,YTrain):
    
    cv = KFold( n_splits =5) 
    result = []

    for train , test in cv.split (XTrain):
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
    return mean_accuracy 

# Transform the data first
dataset= digits

pca = PCA(n_components=200) 
pca.fit(dataset)
pc=pca.components_ # project data to PCA components
projected=dataset.dot(pc.T)

print(crossvalpar(projected,labels)) 

#--------------------------------------------
pca = PCA(n_components=20) 
pca.fit(dataset)
pc=pca.components_ # project data to PCA components
projected=dataset.dot(pc.T)

print(crossvalpar(projected,labels)) 
