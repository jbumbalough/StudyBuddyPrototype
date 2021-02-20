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

##Getting the data
df = pd.read_csv("CourseBuddyDATA_SAMPLE1.csv")
df
scaler = StandardScaler()


# In[2]:


##Transforming the data
dfr = df.replace('A', 5)
dfr = dfr.replace('B', 4)
dfr = dfr.replace('C', 3)
dfr = dfr.replace('D', 2)
dfr = dfr.replace('F', 1)
dfr


# In[3]:


##Predicting final grade, based on all other grades for all students
x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', 'Course Grade'], axis=1), dfr['Course Grade'])


# In[4]:


#x_train #x_test #y_train #y_test


# In[5]:


##Finding the best k neighbors
fe = []
for i in range(1, 13):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    fe.append(np.mean(cross_val_score(knn, x_test, y_test, cv=5)))


# In[6]:


fe


# In[7]:


##The Prediction
k_star = np.argmin(fe)
print(k_star)
knn = KNeighborsClassifier(n_neighbors = k_star, p = 2)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(pred)
knn.score(x_test, y_test)*100


# In[8]:


##Predicting each of the grade items, based on all other grade items for all students
features = ['Class Participation [100]', 'HW#1 [100]', 'HW#2 [100]', 'HW#3 [100]', 'Test#1 [100]', 'Test#2 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']

for i in features:
    print(i)
    x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', i], axis=1), dfr[i])
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    print(y_test)
    print(pred)
    print(knn.score(x_test, y_test)*100)


# In[22]:


##Predicting each of the grade items over time, based on available previous data
#Todo
a_std = pd.DataFrame({'S#' : [100], 'Prerequisite 1': [5], 'Prerequisite 2': [5], 'Prerequisite 3': [5]})
print(a_std)

for i in features:
    pred = knn.predict(a_std)
    print(pred)
    #print(knn.score(x_test, y_test)*100)

#dfr.append(a_std, ignore_index=True)
#print(dfr)


# In[10]:


###
#features = [['Prerequisite 1', 'Prerequisite 2', 'Prerequisite 3', 'Class Participation [100]', 'HW#1 [100]', 'HW#2 [100]', 'HW#3 [100]', 'Test#1 [100]', 'Test#2 [100]', 'Test#3 [100]', 'FinalExam [300]']]
#for feature in features:
#    dfr[feature] = scaler.fit_transform(dfr[feature])

#x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', 'Course Grade'], axis=1), dfr['Course Grade'])


# In[11]:


#fe = []
#for i in range(1, 13):
#    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
#    knn.fit(x_train, y_train)
#    pred = knn.predict(x_test)
#    fe.append(np.mean(cross_val_score(knn, x_test, y_test, cv=5)))


# In[12]:


#k_star = np.argmin(fe)
#print(k_star)
#knn = KNeighborsClassifier(n_neighbors = k_star, p = 2)
#knn.fit(x_train, y_train)
#pred = knn.predict(x_test)
#knn.score(x_test, y_test)*100

