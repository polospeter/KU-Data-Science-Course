
import os

os.chdir('C:\\Users\\Peter')

# Exercise 1 ------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

dataset=np.loadtxt('diatoms.txt')
print(dataset)
np.shape(dataset)

# Plot a single cells ------------------------------------------------------------------------------------------------------
x=dataset[:,0::2]
y=dataset[:,1::2]

x=np.append(x,x[:,0].reshape(-1,1),axis=1)
y=np.append(y,y[:,0].reshape(-1,1),axis=1)

np.shape(x)
np.shape(y)
plt.plot(x[1],y[1])
plt.axis('equal')
plt.title('Plot of a single cell')

#plot all the cells ------------------------------------------------------------------------------------------------------

for i in range(x.shape[0]):
    plt.plot(x[i],y[i])
    plt.axis([-20,20,-20,20])
    plt.axis('equal')
    plt.title('Plot of all the cells')
    
#============================================================================================================================= 
# Exercise 2
 
from sklearn.decomposition import PCA 

dataset=np.loadtxt('diatoms.txt')
pca = PCA(n_components=3) 
pca.fit(dataset)
eig=pca.explained_variance_ #eigenvalues are the same 
print(eig)
eig=np.sqrt(eig)
print(eig) #standard deviation of the data projected onto each of the three PCs 
eigvec=pca.components_
print(eigvec)

m=np.mean(dataset,axis=0) #coloumn wise means

#create arrays
col1=np.empty([3, 180])
col2=np.empty([3, 180])
col3=np.empty([3, 180])
col4=np.empty([3, 180])
col5=np.empty([3, 180])

for i in range(3): 
    col1[i]=m-2*eig[i]*eigvec[i] 
    col2[i]=m-eig[i]*eigvec[i] 
    col3=m
    col4[i]=m+eig[i]*eigvec[i] 
    col5[i]=m+2*eig[i]*eigvec[i]
#-----------------------------------------------------------------------------------------   
dataset=col1

x=dataset[:,0::2]
y=dataset[:,1::2]

x=np.append(x,x[:,0].reshape(-1,1),axis=1)
y=np.append(y,y[:,0].reshape(-1,1),axis=1)  
#-----------------------------------------------------------------------------------------
dataset=col2

x2=dataset[:,0::2]
y2=dataset[:,1::2]

x2=np.append(x2,x2[:,0].reshape(-1,1),axis=1)
y2=np.append(y2,y2[:,0].reshape(-1,1),axis=1)
#-----------------------------------------------------------------------------------------
dataset=m

x5=dataset[0::2]
y5=dataset[1::2]

x5=np.append(x5,x5[0].reshape(-1,1))
y5=np.append(y5,y5[0].reshape(-1,1))
#-----------------------------------------------------------------------------------------
dataset=col4

x3=dataset[:,0::2]
y3=dataset[:,1::2]

x3=np.append(x3,x3[:,0].reshape(-1,1),axis=1)
y3=np.append(y3,y3[:,0].reshape(-1,1),axis=1)
#-----------------------------------------------------------------------------------------
dataset=col5

x4=dataset[:,0::2]
y4=dataset[:,1::2]

x4=np.append(x4,x4[:,0].reshape(-1,1),axis=1)
y4=np.append(y4,y4[:,0].reshape(-1,1),axis=1)
#-----------------------------------------------------------------------------------------
# Plot- Principal component 1

blues= plt.get_cmap('Blues')

plt.plot(x[0],y[0],color=blues(0))
plt.plot(x2[0],y2[0],color=blues(0.2))
plt.plot(x5,y5,color=blues(0.4))
plt.plot(x3[0],y3[0],color=blues(0.6))
plt.plot(x4[0],y4[0],color=blues(0.8))
plt.title('Temporal development of PC1')
# ===================================================================================================================
#   do it for the other PCA as well
# ===================================================================================================================
# Plot- Principal component 2

blues= plt.get_cmap('Greens') 
plt.plot(x[1],y[1],color=blues(0.4))
plt.plot(x2[1],y2[1],color=blues(0.5))
plt.plot(x5,y5,color=blues(0.6))
plt.plot(x3[1],y3[1],color=blues(0.7))
plt.plot(x4[1],y4[1],color=blues(0.8))
plt.title('Temporal development of PC2')

#------------------------------------------------------------------------------------------------------------------------

# Plot- Principal component 3

blues= plt.get_cmap('Reds') 
plt.plot(x[1],y[1],color=blues(0.2))
plt.plot(x2[1],y2[1],color=blues(0.4))
plt.plot(x5,y5,color=blues(0.6))
plt.plot(x3[1],y3[1],color=blues(0.8))
plt.plot(x4[1],y4[1],color=blues(1))
plt.title('Temporal development of PC3')

#==========================================================================================================================
# Exercise 3
data=np.loadtxt('pca_toydata.txt')
pca = PCA(n_components=2) 
pca.fit(data)
eig=pca.explained_variance_ #eigenvalues are the same 
pc=pca.components_

#project the datapoints
signals=data.dot(pc.T)  # my implementation
plt.scatter(signals[:, 0], signals[:, 1], alpha=0.3) 
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Projection of toy dataset')

#-----------------------------------------------------------------------------------------------------------------------
# projection excluding last two data points
data2=data[0:100] 
pca = PCA(n_components=2) 
pca.fit(data2)
eig=pca.explained_variance_ #eigenvalues are the same 
pc=pca.components_

X_projec2 = pca.fit_transform(data2)

plt.scatter(X_projec2[:, 0], X_projec2[:, 1], alpha=0.3)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Projection of toy dataset excluding last 2 points')

#=====================================================================================================================
# Exercise 4

#import data
test_data=np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
train_data=np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')

x_train=train_data[:,:-1]
y_train=train_data[:,-1]
x_test=test_data[:,:-1]
y_test=test_data[:,-1]
#-----------------------------------------------------------

# Built-in PCA
from sklearn.cluster import KMeans 
startingPoint=np.vstack((x_train[0,],x_train[1,])) #initializing
kmeans=KMeans(n_clusters=2,n_init=1,init=startingPoint).fit(x_train)
#kmeans=KMeans(n_clusters=2).fit(x_train)
print(kmeans.cluster_centers_)

pca = PCA(n_components=2)
pca.fit(x_train)
pcc=pca.components_ #eigenvectors
X_projec = pca.fit_transform(x_train)

centress=(kmeans.cluster_centers_).dot(pcc.T) #cluster centers in 2D

#MY implementation ----------------------------------------------------------------------------------------------------

from sklearn.metrics import pairwise_distances_argmin

def findclusters(X, n_clusters):
    #initial centers with the first two data points
    centers = X[0:n_clusters]
 
    while True:
        # calculate distance of all data points from all centers
        # and give them labels according to the closest cluster center
        labels = pairwise_distances_argmin(X, centers)
        
        # centroid recalculation: taking the mean of all data points again
        new_centers = np.array([X[labels == i].mean(axis=0)
                                for i in range(n_clusters)])
        # do it until convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

#-----------------------------------------------------------------------------------------------------------------

centers, labels = findclusters(x_train, 2)

centres=(kmeans.cluster_centers_).dot(pcc.T) 
centress=(centers).dot(pcc.T)
print(centress) #cluster centers in 2D

#Plot
labels=labels.reshape((1000,1))

labda=np.concatenate((labels,X_projec),axis=1)

cluster0=(labda[:,0]==0)
cluster1=(labda[:,0]==1)

data0=X_projec[cluster0]
data1=X_projec[cluster1]

#Color each point according to its center
plt.scatter(data0[:, 0], data0[:, 1], alpha=0.1,color='Green')
plt.scatter(data1[:, 0], data1[:, 1], alpha=0.1,color='Red')
plt.scatter(centress[0,0],centress[0,1],color='Green',s=100)
plt.scatter(centress[1,0],centress[1,1],color='Red',s=100)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Find cluster centers and then do PCA')

# Look for cluster centres in the new transformed data ---------------------------------------------------------------------

X_projec = pca.fit_transform(x_train)
centers, labels = findclusters(X_projec, 2)
print(centers) 

labels=labels.reshape((1000,1))

labda=np.concatenate((labels,X_projec),axis=1)

cluster0=(labda[:,0]==0)
cluster1=(labda[:,0]==1)

data0=X_projec[cluster0]
data1=X_projec[cluster1]

#Color each point according to its center
plt.scatter(data0[:, 0], data0[:, 1], alpha=0.1,color='Green')
plt.scatter(data1[:, 0], data1[:, 1], alpha=0.1,color='Red')
plt.scatter(centers[0,0],centers[0,1],color='Green',s=100)
plt.scatter(centers[1,0],centers[1,1],color='Red',s=100)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Do PCA and then find cluster centers')
# as we can see this way, we get much reasonable cluster centers