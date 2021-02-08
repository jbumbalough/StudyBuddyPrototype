#!/usr/bin/env python
# coding: utf-8

# In[1]:


###
# Study Buddy
# Coursebuddy Prototype
# A basic prototype for the Coursebuddy system.
###

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("CourseBuddyDATA_SAMPLE1.csv")
df
scaler = StandardScaler()


# In[2]:


dfr = df.replace('A', 5)
dfr = dfr.replace('B', 4)
dfr = dfr.replace('C', 3)
dfr = dfr.replace('D', 2)
dfr = dfr.replace('F', 1)
dfr


# In[3]:


x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', 'Course Grade'], axis=1), dfr['Course Grade'])


# In[4]:


x_train


# In[5]:


x_test


# In[6]:


y_train


# In[7]:


y_test


# In[8]:


fe = []
for i in range(1, 13):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    fe.append(np.mean(cross_val_score(knn, x_test, y_test, cv=5)))


# In[9]:


fe


# In[10]:


k_star = np.argmin(fe)
print(k_star)
knn = KNeighborsClassifier(n_neighbors = k_star, p = 2)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
knn.score(x_test, y_test)*100


# In[11]:


################################################
features = [['Prerequisite 1', 'Prerequisite 2', 'Prerequisite 3', 'Class Participation [100]', 'HW#1 [100]', 'HW#2 [100]', 'HW#3 [100]', 'Test#1 [100]', 'Test#2 [100]', 'Test#3 [100]', 'FinalExam [300]']]
for feature in features:
    dfr[feature] = scaler.fit_transform(dfr[feature])

x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', 'Course Grade'], axis=1), dfr['Course Grade'])


# In[12]:


fe = []
for i in range(1, 13):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    fe.append(np.mean(cross_val_score(knn, x_test, y_test, cv=5)))


# In[13]:


k_star = np.argmin(fe)
print(k_star)
knn = KNeighborsClassifier(n_neighbors = k_star, p = 2)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
knn.score(x_test, y_test)*100

