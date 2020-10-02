
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
    
# Exercise 1 ------------------------------------------------------------------------------------------------------    
#---------IMPORT DATA--------------------------------------------------------------------------
murder=np.loadtxt('murderdata2d.txt')

test_data=np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
train_data=np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')

x_train=train_data[:,:-1]
y_train=train_data[:,-1]
x_test=test_data[:,:-1]
y_test=test_data[:,-1]

# Principal component analysis-own implementation -------------------------------------------------------------
def princip(data):
    mn = np.mean(data,axis=0)
    data2 = data - mn #centralized
    
    #calculate the covariance matrix
    covar=np.cov(data2.T) #it is the transpose
    
    #calculating eigen values and vectors
    eigenvalues, eigenvectors =np.linalg.eig(covar)
        
    #sort the elements-- the variances are in percentages now
    sorteig=sorted(eigenvalues,reverse=True)
    rindices=np.argsort(-eigenvalues) #indices  
    pc = eigenvectors[:,rindices.T]
    
    signals=data2.dot(pc) #the transformed dataset
    
    return (sorteig, pc,signals)

print(princip(murder))

prin=princip(murder)
prin[1]

#Comparison
pca = PCA(n_components=2)
pca.fit(murder)

print(prin[0])
print(pca.explained_variance_) #eigenvalues are the same

print(pca.components_)
print(prin[1])  #here we can see that the two indeed bring the same results,except with opposite sign for each element
# meaning that the vectors are still on the same line,simply they point to opposite directions

# Part b ----Scatter plot ---------------------------------------------------------------------------------------

# Compute the corresponding standard deviations
data=murder
s0 = np.sqrt(pca.explained_variance_[0])
s1 = np.sqrt(pca.explained_variance_[1])

prin=princip(murder)
evecs=-prin[1] # i got eigenvectors pointing to the opposite direction compared to the ones from the built in PCA
plt.scatter(data[:,0],data[:,1])
plt.plot([np.mean(data[:,0]),np.mean(data[:,0])+ s0*evecs[0,0]], [np.mean(data[:,1]), np.mean(data[:,1])+s0*evecs[1,0]], 'r')
plt.plot([np.mean(data[:,0]),np.mean(data[:,0])+ s1*evecs[0,1]], [np.mean(data[:,1]), np.mean(data[:,1])+s1*evecs[1,1]], 'r')
plt.scatter(np.mean(data[:,0]),np.mean(data[:,1]),facecolors='red',alpha=.55, s=100) #mean point
plt.xlabel('Murder feature 1')
plt.ylabel('Murder feature 2')

# Part c -------------------------------------------------------------------------------------------------------------

plt.plot(princip(x_train)[0])
plt.xlabel('Principal component index')
plt.ylabel('Variance')

pca2 = PCA(n_components=13)
pca2.fit(x_train) #gives the same results

#--------------
eigenvalues=princip(x_train)[0]
eig=eigenvalues/np.sum(eigenvalues)   

x=range(1,14)
plt.plot(x,np.cumsum(eig))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

print(np.cumsum(eig))
#-----------------------------------------------------------------------------------------------------------------------
# EXERCISE 2 -----------------------------------------------------------------------------------------------------------

# own implementation
def princip2(data,n):
    mn = np.mean(data,axis=0)
    data2 = data - mn #centralized
    
    #calculate the covariance matrix
    covar=np.cov(data2.T) #it is the transpose
    
    #calculating eigen values and vectors
    eigenvalues, eigenvectors =np.linalg.eig(covar)
        
    #sort the elements in decreasing order
    sorteig=sorted(eigenvalues,reverse=True)
    rindices=np.argsort(-eigenvalues) #indices  
    pc = eigenvectors[:,rindices.T]
    pc=pc[:,0:n] # n is the number of PC we need
    
    signals=data2.dot(pc) #the transformed dataset
    
    return (sorteig, pc,signals)


project=princip2(x_train,2)[2]
plt.scatter(project[:, 0], project[:, 1])
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Projected data with my implementation')

pca = PCA(n_components=2)
pca.fit(x_train)
X_projec = pca.fit_transform(x_train)
plt.scatter(X_projec[:, 0], X_projec[:, 1], alpha=0.3)
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.title('Projected data with built in PCA')
# as we can see my implementation is results in the inverse of the image we got with the built in PCA,
# which is not surprising since in the previous exercise I have pointed out, that my eigenvectors point to the opposite direction
# compared to the ones in the built in PCA

# EXERCISE 3 -----------------------------------------------------------------------------

#---------IMPORT DATA--------------------------------------------------------------------------
test_data=np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
train_data=np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')

x_train=train_data[:,:-1]
y_train=train_data[:,-1]
x_test=test_data[:,:-1]
y_test=test_data[:,-1]

#--------My implementation -----------------------------------------------------------------------

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

# I will use the train dataset as the pesticide data
findclusters(x_train, 2)

centers, labels = findclusters(x_train, 2)
print(centers) #gives the same results as the built-in version

from sklearn.cluster import KMeans 

# Comparison - they indeed produce the same results
startingPoint=np.vstack((x_train[0,],x_train[1,])) #initializing
kmeans=KMeans(n_clusters=2,n_init=1,init=startingPoint).fit(x_train)
print(kmeans.cluster_centers_)
print(centers)
