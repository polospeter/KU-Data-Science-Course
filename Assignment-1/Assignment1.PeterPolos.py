
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

dataset=np.loadtxt('smoking.txt')

# Exercise 1 #=================================================================================================================
    
#masks
def meanFEV1(data):
    smoker_mask=(data[:,4]==1)
    non_smoker_mask=(data[:,4]==0)

    fev_smoke=data[smoker_mask,1]
    fev_nonsmoke=data[non_smoker_mask,1]

    a=np.mean(fev_smoke)
    b=np.mean(fev_nonsmoke)

    return a,b

print('The mean FEV1 of smokers:', meanFEV1(dataset)[0],'\nThe mean FEV1 of non-smokers:', meanFEV1(dataset)[1])

# Exercise 2 #=================================================================================================================
    smoker_mask=(dataset[:,4]==1)
    non_smoker_mask=(dataset[:,4]==0)

    fev_smoke=dataset[smoker_mask,1]
    fev_nonsmoke=dataset[non_smoker_mask,1]
    
    #smokers
    plt.subplot(131)
    plt.boxplot(fev_smoke)
    plt.xlabel("smokers") #give title to x axis
    plt.ylabel("FEV1 value")
    plt.title('Boxplot of FEV1 value of smokers')
    
    #non-smokers
    plt.subplot(133)
    plt.boxplot(fev_nonsmoke)
    plt.xlabel("Non-smokers") #give title to x axis
    plt.ylabel("FEV1 value")
    plt.title('Boxplot of FEV1 value of non-smokers')


# Exercise 3 #=================================================================================================================
def hyptest(data):
    smoker_mask=(data[:,4]==1)
    non_smoker_mask=(data[:,4]==0)
    fev_smoke=data[smoker_mask,1]
    fev_nonsmoke=data[non_smoker_mask,1]
    t=stats.ttest_ind(fev_smoke,fev_nonsmoke)[0]
    p=stats.ttest_ind(fev_smoke,fev_nonsmoke)[1]
    if p<0.05:
        result='Reject'
    else:
        result='Accept'
    return result,t,p

print('Hypothesis:',hyptest(dataset)[0],'\np-value:',hyptest(dataset)[2],'\nT-statistic:',hyptest(dataset)[1])

# Exercise 4 #=================================================================================================================
    plt.scatter(dataset[:,0],dataset[:,1])
    plt.xlabel("Age") 
    plt.ylabel("FEV1")
    plt.title("Scatter plot between age and FEV1 value")

    corr=np.corrcoef(dataset[:,0],dataset[:,1])
print('The correlation between age and FEV1 value is',corr[0,1])

# Exercise 5 #=================================================================================================================
    np.histogram
    smoker_mask=(dataset[:,4]==1)
    non_smoker_mask=(dataset[:,4]==0)

    smoke_age=dataset[smoker_mask,0]
    nonsmoke_age=dataset[non_smoker_mask,0]
    
    plt.subplot(121)
    plt.hist(smoke_age)
    plt.title("Histogram of smokers")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.subplot(122)
    plt.hist(nonsmoke_age)
    plt.title("Histogram of non-smokers")
    plt.xlabel("Age")
