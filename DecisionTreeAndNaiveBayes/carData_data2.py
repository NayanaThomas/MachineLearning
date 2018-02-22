#!/usr/bin/python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np


# In[19]:


#reading the input data from the csv file
INPUT_PATH="carTrainData.csv"


# In[20]:


#header values are defined.
headers = ["buying price", "maint cost", "doors", "persons", "lug_boot", "safety", "v7"]


# In[21]:


dataset = pd.read_csv(INPUT_PATH, header=None, names=headers, na_values="?" )
dataset = dataset.iloc[1:]


# In[22]:


dataset


# In[23]:


#here the feature values are converted to categorical values
for header in headers:
    dataset[header]=dataset[header].astype("category")
    dataset[header]=dataset[header].cat.codes    


# In[24]:


#the first 6 column values are the feature values
training_array=dataset.iloc[:,0:6].values
#arr1.dtype


# In[41]:


#the last column values are the target attribute values
target_array=dataset.iloc[:,6:7].values
#list(map(str, np.unique(target_array)))


# In[27]:


#importing the decision tree classifier from the sklearn library
from sklearn import tree 
classifier = tree.DecisionTreeClassifier(criterion="entropy")
classifier.fit(training_array, target_array)


# In[34]:


#import graphviz


# In[30]:


#dot_data = tree.export_graphviz(classifier, out_file=None)


# In[35]:


#graph = graphviz.Source(dot_data)
#dot_data = tree.export_graphviz(classifier, out_file=None, 
#                         feature_names=headers[0:6],  
#                         class_names=list(map(str, np.unique(target_array))),  
#                         filled=True, rounded=True,  
#                         special_characters=True)


# In[36]:


#graph = graphviz.Source(dot_data)
#graph


# In[46]:


#we are getting the target feature values predicted by the classifier
training_prediction = classifier.predict(training_array)


# In[48]:


#calculating the Accuracy of the classifier for the train data
from sklearn.metrics import accuracy_score
train_accuracy_score = accuracy_score(target_array, training_prediction)
print("Train accuracy score: ", train_accuracy_score)


# In[51]:


#calculating the confusion matrix of the classifier for the train data
from sklearn.metrics import confusion_matrix
train_confusion_matrix = confusion_matrix(target_array, training_prediction)
print( "Training Confusion matrix: \n", train_confusion_matrix)


# In[77]:


#reading the test data from the file
testDataSet = pd.read_csv("carTestData.csv", header=None, names=headers, na_values="?")
testDataSet = testDataSet[1:]
#here the feature values are converted to categorical values
for header in headers:
    testDataSet[header]=testDataSet[header].astype("category")
    testDataSet[header]=testDataSet[header].cat.codes   


# In[78]:


#we are splitting the testDataSet to obtain the feature and target data
test_feature = testDataSet.iloc[:,0:6]
test_target = testDataSet.iloc[:,6:7]


# In[85]:


#we are predicting the target attribute values using the classifier
test_prediction = classifier.predict(test_feature)


# In[86]:


#calculating the Accuracy of the classifier for the test data
test_accuracy_score = accuracy_score(test_target, test_prediction)
print("Test accuracy score: ", test_accuracy_score)


# In[88]:


#calculating the confusion matrix of the classifier for the test data
from sklearn.metrics import confusion_matrix
test_confusion_matrix = confusion_matrix(test_target, test_prediction)
print( "Test Confusion matrix: \n", test_confusion_matrix)

