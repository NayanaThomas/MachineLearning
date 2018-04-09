
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# In[2]:


#reading test and train data
datatrain = pd.read_csv('bikeRentalHourlyTrain.csv', header=0, na_values=np.NaN, sep=',')
datatest = pd.read_csv('bikeRentalHourlyTest.csv', header=0, na_values=np.NaN, sep=',')


# In[3]:


#removing unwanted columns
datatrain = datatrain.drop(datatrain.columns[[0, 1, 2, 11, 15, 16]], axis=1)
datatest = datatest.drop(datatest.columns[[0, 1, 2, 11, 15, 16]], axis=1)


# In[4]:


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
#calling the methods on train and test data
cleanDataFrame(datatrain)
cleanDataFrame(datatest)


# In[5]:


#method to create dummy variables
def changeToCategoryType(data):
    for header in data.columns:
        #if header!='cnt':
            data[header] = data[header].astype('category').cat.codes
#calling the methods on train and test data
changeToCategoryType(datatest)
changeToCategoryType(datatrain)            


# In[6]:


#Scaling the input data
#makes the algorithm converge faster
#We use a built in function to do this.
scaler = StandardScaler()
#Splitting original train data to feature list and target list.
X_train = datatrain.iloc[:,0:11]
y_train = datatrain.iloc[:,11:12].iloc[:,0]
scaler.fit(X_train)
#Splitting original test data to feature list and target list.
X_test = datatest.iloc[:,0:11]
y_test = datatest.iloc[:,11:12].iloc[:,0]


# In[7]:


# Now use the scaler to scale both test and train predictors
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[18]:


#method to build neural network and print MSE of both train and test data
def createNeuralNetwork():
    print('Neural Network Metrics')
    #Here we configure the architecure. These are hidden layers only
    #The function will automatically create input nodes (one for each variable) and 
    #one output node (for the target value)
    MLPregr = MLPRegressor(hidden_layer_sizes=(15, 15, 15), max_iter=2000)
    #Fit the model and learn the weights
    MLPregr.fit(X_train,y_train)
    #Let's now use the model to predict the target:
    predicted_train = MLPregr.predict(X_train)
    predicted_test = MLPregr.predict(X_test)
    print("Training Mean squared error: %.2f" % mean_squared_error(y_train, predicted_train))
    print("Test Mean squared error: %.2f" % mean_squared_error(y_test, predicted_test))
    
    #5-fold cross validation for test data set
    #We provide train/test indices to split data in test sets. 
    #We split the dataset into 5 consecutive folds (without shuffling by default).
    #Each fold is then used once as a validation while the 4 remaining folds form the training set.
    print("Metrics after 5-fold CV")
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_train)
    MSE = 0.0
    for train_index, test_index in kf.split(X_test):
        MLPregr.fit(X_test[train_index],y_test[train_index])
        predictions = MLPregr.predict(X_test[test_index])
        MSE += sum((predictions - y_test[test_index])**2)/len(predictions)
    print("Test Mean squared error: %.2f" % (MSE/5.0))


# In[ ]:


#creating the neural network for the data set
createNeuralNetwork()


# In[16]:


#method to create linear regression model and print MSE of test and train data.
def createRegression():
    print('Linear Regression Metrics:')
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    predicted_train = regr.predict(X_train)
    print("Training Mean squared error: %.2f" % mean_squared_error(y_train, predicted_train))
    predicted_test = regr.predict(X_test)
    print("Test Mean squared error: %.2f" % mean_squared_error(y_test, predicted_test))
    
    #5-fold cross validation.
    #We provide train/test indices to split data in test sets. 
    #We split the dataset into 5 consecutive folds (without shuffling by default).
    #Each fold is then used once as a validation while the 4 remaining folds form the training set. 
    print("Metrics after 5-fold CV")
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_test)
    MSE = 0.0
    for train_index, test_index in kf.split(X_test):
        regr.fit(X_test[train_index],y_test[train_index])
        predictions = regr.predict(X_test[test_index])
        MSE += sum((predictions - y_test[test_index])**2)/len(predictions)
    print("Test Mean squared error: %.2f" % (MSE/5.0))


# In[17]:


#creating the linear regression model for the data set
createRegression()


# In[12]:


# Finally, let's try Ridge and Lasso. We'll do Ridge first.
regrRidge = linear_model.Ridge(alpha = 1)
regrRidge.fit(X_train, y_train)
ridgePredictions = regrRidge.predict(X_test)
# The mean squared error
print('Linear Rigression Ridge Metrics')
print("Mean squared error: %.2f"% mean_squared_error(y_test, ridgePredictions))


# In[13]:


# Not much of difference - seems all variables are important. We try Lasso ...
regrLasso = linear_model.Lasso(alpha = .1)
regrLasso.fit(X_train, y_train)
lassoPredictions = regrLasso.predict(X_test)
# The mean squared error
print('Linear Regression Lasso Metrics')
print("Mean squared error: %.2f"% mean_squared_error(y_test, lassoPredictions))


# In[14]:


#method to build Radial SVM and print its mean squared error of both train and test data
def createKNN():
    print('KNN with k=8 Metrics')
    KNN = KNeighborsRegressor(n_neighbors=8)
    KNN.fit(X_train,y_train)
    predicted_train = KNN.predict(X_train)
    predicted_test = KNN.predict(X_test)
    print("Training Mean squared error: %.2f" % mean_squared_error(y_train, predicted_train))
    print("Test Mean squared error: %.2f" % mean_squared_error(y_test, predicted_test))  
    
    #5-fold cross validation.
    #We provide train/test indices to split data in test sets. 
    #We split the dataset into 5 consecutive folds (without shuffling by default).
    #Each fold is then used once as a validation while the 4 remaining folds form the training set.
    print("Metrics after 5-fold CV")
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_test)
    MSE = 0.0
    for train_index, test_index in kf.split(X_test):
        KNN = KNeighborsRegressor(n_neighbors=4)
        KNN.fit(X_test[train_index],y_test[train_index])
        predictions = KNN.predict(X_test[test_index])
        MSE += sum((predictions - y_test[test_index])**2)/len(predictions)
    print("Test Mean squared error: %.2f" % (MSE/5.0))


# In[15]:


createKNN()

