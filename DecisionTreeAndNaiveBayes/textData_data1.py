#!/usr/bin/python
# coding: utf-8

# In[36]:


#required python packages are imported
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import re


# In[37]:


#read train data from the csv file
training_dataset = pd.read_csv("textTrainData.txt", sep="\t", header=0, encoding='latin1')


# In[38]:


#we are splitting the text based on the sentiment
def getTextForSentiment(sentence, sentiment):
    # Join together the text in the reviews for a particular sentiment.
    # We lowercase to avoid "Not" and "not" being seen as different words, for example.
    s=""
    for index, row in sentence.iterrows():
        if(row["Sentiment"]==sentiment):
            s=s+row["Sentence"].lower()
    return s        


# In[39]:


#we are splitting dataset to positive and negative samples
positive_sample = getTextForSentiment(training_dataset, 1)
negative_sample = getTextForSentiment(training_dataset, 0)


# In[40]:


#we are defining a function that will count the number of words for each sample
def countWords(Sentence):
    words = re.split("\s+", Sentence)
    return Counter(words)


# In[41]:


#we are counting the number of words in the positive and negative samples in the dataset
count_positive = countWords(positive_sample)
count_negative = countWords(negative_sample)


# In[42]:


#next method will calculate the number of samples with the given sentiment
def countSampleForSentiment(sentiment):
    c=0
    for index, row in training_dataset.iterrows():
        if(row["Sentiment"]==sentiment):
            c+=1
    return c        


# In[43]:


#counting the number of samples with both sentiment values
positive_sample_count = countSampleForSentiment(1)
negative_sample_count = countSampleForSentiment(0)


# In[44]:


#we are calculating the probabilities of positive and negative samples
positive_prob = positive_sample_count/len(training_dataset)
negative_prob = negative_sample_count/len(training_dataset)


# In[45]:


#we are creating a function that will calculate the proability for a sample sentence
def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob


# In[47]:


def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, count_negative, negative_prob, negative_sample_count)
    positive_prediction = make_class_prediction(text, count_positive, positive_prob, positive_sample_count)
    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return 0
    return 1


# In[48]:


#predicting the target values for the Sentence using the given classifier
training_predictions = [make_decision(row['Sentence'], make_class_prediction) for index,row in training_dataset[0:len(training_dataset)].iterrows()]


# In[49]:


#actual target attribute values for training data are retrieved here
training_actual = training_dataset['Sentiment'].tolist()


# In[50]:


#we are calculating the accuracy for the training data here
training_accuracy = sum(1 for i in range(len(training_predictions)) if training_predictions[i] == training_actual[i]) / float(len(training_predictions))
print("Training Accuracy: {0:.4f}".format(training_accuracy))


# In[51]:


#we are calculating the confusion matrix for the training data below
from sklearn.metrics import confusion_matrix
training_confusion_matrix = confusion_matrix(training_actual, training_predictions)
print("Training Confusion Matrix: \n", training_confusion_matrix)


# In[58]:


#we are reading the test data here
test_dataset = pd.read_csv("textTestData.txt", sep="\t", header=0, encoding='latin1')


# In[63]:


#predicting the target values for the Sentence using the given classifier for test data
test_predictions = [make_decision(row['Sentence'], make_class_prediction) for index,row in test_dataset[0:len(test_dataset)].iterrows()]


# In[64]:


#actual target attribute values for test data are retrieved here
test_actual = test_dataset['Sentiment'].tolist()


# In[65]:


#we are calculating the accuracy for the test data here
test_accuracy = sum(1 for i in range(len(test_predictions)) if test_predictions[i] == test_actual[i]) / float(len(test_predictions))
print("Test Accuracy: {0:.4f}".format(test_accuracy))


# In[66]:


#we are calculating the confusion matrix for the test data below
test_confusion_matrix = confusion_matrix(test_actual, test_predictions)
print("Training Confusion Matrix: \n", test_confusion_matrix)

