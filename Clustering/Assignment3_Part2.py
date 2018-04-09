#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np


# In[147]:


dataset = pd.read_csv('dataset_Facebook.csv',sep=';')


# In[148]:


#method to clean the dataset
#based on the datatype of the attribute, the NaN values are replaced with standard values
#if datatype is object, replaced with empty string
#if datatype is int64, replaced with rounded off mean value
#if datatype is float64, replaced with meanvalue
def cleanDataFrame(data):
    for i in range(0, data.shape[1]):
        colValues = data.iloc[:,i:i+1].iloc[:,0]
        colType = data.iloc[:,i:i+1].iloc[:,0].dtype
        if colType == 'int64':
            data.iloc[:,i:i+1] = data.iloc[:,i:i+1].fillna(round(colValues.mean()))
        elif colType == 'float64':
            data.iloc[:,i:i+1] = data.iloc[:,i:i+1].fillna(colValues.mean())        
        elif colType == 'object':
            data.iloc[:,i:i+1] = data.iloc[:,i:i+1].fillna('')        


# In[149]:


#cleaning both test and train data by calling the defined function
cleanDataFrame(dataset)


# In[150]:


for header in list(dataset):
    if(dataset[header].dtypes=='O'):
        dataset[header] = dataset[header].astype('category').cat.codes


# In[151]:


centroid = dataset.mean().to_frame().T
centroid


# In[152]:


#Methood to find the total sum of squares for the dataset D and centroid C
def findTSSforD(D, C):
    mTSS = 0
    for index, row in D.iterrows():
        rowsum = 0
        for header in list(D):   
            rowsum += math.pow((row[header]-C[header]), 2)                 
        mTSS += rowsum
    return mTSS    


# In[153]:


TSS = findTSSforD(dataset, centroid)
print(TSS)


# In[154]:


print('K-Means Clustering')
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataset)
    kTSS = [None]*k
    dataset['cluster'] = kmeans.labels_
    for clustNum in range(0, k):
        centroidForClust = pd.DataFrame(kmeans.cluster_centers_[clustNum]).T.iloc[:,0:19]             
        centroidForClust.columns = list(dataset.iloc[:,0:19])        
        kTSS[clustNum] = findTSSforD(dataset[dataset['cluster']==clustNum].iloc[:,0:19], centroidForClust)
    TWSS = sum(kTSS) 
    print('For k=', k, ' total within sum of squares/total sum of squares=', TWSS/TSS)     


# In[155]:


print('H-Clustering')
for k in range(1, 11):
    hcluster = AgglomerativeClustering(n_clusters=k)
    hcluster.fit(dataset)
    kTSS = [None]*k
    dataset['cluster'] = hcluster.labels_
    for clustNum in range(0, k):        
        kTSS[clustNum] = findTSSforD(dataset[dataset['cluster']==clustNum].iloc[:,0:19], dataset[dataset['cluster']==clustNum].iloc[:,0:19].mean().to_frame().T)
    TWSS = sum(kTSS)
    print('For k=', k, ' total within sum of squares/total sum of squares=', TWSS/TSS)


# In[156]:


print('Gaussian Mixture Models')
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, reg_covar=1e-4)
    gmm.fit(dataset)
    #print(gmm.means_)
    kTSS = [None]*k
    dataset['cluster'] = gmm.predict(dataset)
    for clustNum in range(0, k):        
        kTSS[clustNum] = findTSSforD(dataset[dataset['cluster']==clustNum].iloc[:,0:19], dataset[dataset['cluster']==clustNum].iloc[:,0:19].mean().to_frame().T)
    TWSS = sum(kTSS) 
    print('For k=', k, ' total within sum of squares/total sum of squares=', TWSS/TSS)

