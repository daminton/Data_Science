#!/usr/bin/env python
# coding: utf-8

# Load the dataset from sklearn

# In[2]:


from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',version=1)
mnist.keys()


# Let's learn more about the data...

# In[3]:


mnist.DESCR


# The data consists of two important keys: data and target. 
# Data - contains an array with one row per instance and one column per feature
# Target - contains an array with the labels (we're classifying information so labels are really important)

# In[4]:


data, labels = mnist['data'], mnist['target']


# Let's see what shapes we're dealing with...

# In[5]:


data.shape


# In[6]:


labels.shape


# Let's look at one digit from the dataset. Each image has 784 features - 28x28. Let's grab and image's data, reshape it, and then display it. 

# In[7]:


import matplotlib as mpl
import matplotlib.pyplot as plt
digit = data[0]
digit_image = digit.reshape(28,28)
plt.imshow(digit_image, cmap='binary')
plt.axis('off')
plt.show()


# In[8]:


labels[0]


# The image kind of looks like a 5, and sure enough it is labeled as a 5. Because ML algorithms usually expect numbers, we should cast the labels as numbers.

# In[10]:


import numpy as np
labels = labels.astype(np.uint8)


# Before we start doing any manipulations, we should create our training set and test set. What does the description have to say about the dataset in regards to training/test?

# In[22]:


data_train, data_test, labels_train, 
labels_test = data[:60000], data[60000:], labels[:60000], 
labels[60000:]


# Break of the data based on fives and not fives

# In[14]:


labels_train_5 = (labels_train == 5)
labels_test_5 = (labels_test == 5)


# Back to notes...

# Let's use a Stochastic Gradient Descent algorithm to train on identifying fives

# In[16]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 21)


# Why random_state = 21?

# In[17]:


sgd_clf.fit(data_train, labels_train_5)


# Model is trained...now, we can use it to predict the value of a digit

# In[18]:


sgd_clf.predict([digit])


# This value was a five if you remember from before so we received what we expected. 
# Back to notes...

# In[19]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, data_train, labels_train_5, cv = 3, scoring = 'accuracy')


# Wow! Well done!!! Let's call it a day!

# In[20]:


from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, data, labels=None):
        return self
    def predict(self, data):
        return np.zeros((len(X), 1), dtype = bool)


# In[21]:


never_5_clf = Never5Classifier()


# In[ ]:


cross_val_score(never_5_)

