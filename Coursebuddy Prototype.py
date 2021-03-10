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
features = ['Class Participation [100]', 'HW#1 [100]', 'Test#1 [100]', 'HW#2 [100]', 'Test#2 [100]', 'HW#3 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']

for i in features:
    print(i)
    x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', i], axis=1), dfr[i])
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    print(y_test)
    print(pred)
    print(knn.score(x_test, y_test)*100)


# In[9]:


def acc(obs, accept):
    return abs(((obs - accept) / accept) * 100)


# In[10]:


def predict(original):
    copy = original.copy()
    features = ['Class Participation [100]', 'HW#1 [100]', 'Test#1 [100]', 'HW#2 [100]', 'Test#2 [100]', 'HW#3 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']
    for i in features:
        print("Feature: ", i)
        x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', i], axis=1), dfr[i])
        x_test = copy.drop(['S#', i], axis=1)
        #y_test = a_final[i]
        #print("x: ", x_test)
        #print("y: ", y_test)
        knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        print("Prediction: ", pred)
        #for j in a_final[i]:
            #print("Expected: ", j)
            #print("PE: ", acc(pred, j))
        copy[i] = pred
        print("\n===============\n")


# In[11]:


a_original = pd.DataFrame({'S#' : [100], 'Prerequisite 1': [5], 'Prerequisite 2': [5], 'Prerequisite 3': [5], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
predict(a_original)


# In[12]:


b_original = pd.DataFrame({'S#' : [101], 'Prerequisite 1': [4], 'Prerequisite 2': [4], 'Prerequisite 3': [4], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
predict(b_original)


# In[13]:


class GradeItem:
    def __init__(self, p, q, t):
        self.points = p
        self.questions = q
        self.topics = t


# In[14]:


h1 = GradeItem(100, ["q1", "q2", "q3"], ["t1"])
print(h1.points)
print(h1.questions)
print(h1.topics)


# In[15]:


new_stud = pd.DataFrame({'S#' : [102], 'Prerequisite 1': [5], 'Prerequisite 2': [4], 'Prerequisite 3': [3], 'Class Participation [100]' : [0], GradeItem(100, ["q1", "q2", "q3"], ["t1"]) : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
new_stud


# In[16]:


#Recommending similar students
x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#'], axis=1), dfr['S#'])
knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(y_test)
print(pred)
print(knn.score(x_test, y_test)*100)


# In[17]:


##Predicting each of the grade items over time, based on available previous data
a_original = pd.DataFrame({'S#' : [100], 'Prerequisite 1': [5], 'Prerequisite 2': [5], 'Prerequisite 3': [5], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
a_copy = a_original.copy()
a_final = pd.DataFrame({'S#' : [100], 'Prerequisite 1': [5], 'Prerequisite 2': [5], 'Prerequisite 3': [5], 'Class Participation [100]' : [100], 'HW#1 [100]' : [90], 'HW#2 [100]' : [90], 'HW#3 [100]' : [90], 'Test#1 [100]' : [90], 'Test#2 [100]' : [90], 'Test#3 [100]' : [90], 'FinalExam [300]' : [270], 'Course Grade' : [5]})
#print(a_std)
#print(a_final)

for i in features:
    print("Feature: ", i)
    x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', i], axis=1), dfr[i])
    x_test = a_original.drop(['S#', i], axis=1)
    y_test = a_final[i]
    print("x: ", x_test)
    print("y: ", y_test)
    knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    print("Prediction: ", pred)
    for j in a_final[i]:
        print("Expected: ", j)
        print("PE: ", acc(pred, j))
    a_copy[i] = pred
    print("\n===============\n")
    
print(a_original)
print("\n===============\n")
print(a_copy)
print("\n===============\n")
print(a_final)

#dfr.append(a_std, ignore_index=True)
#print(dfr)


# In[18]:


###
#features = [['Prerequisite 1', 'Prerequisite 2', 'Prerequisite 3', 'Class Participation [100]', 'HW#1 [100]', 'HW#2 [100]', 'HW#3 [100]', 'Test#1 [100]', 'Test#2 [100]', 'Test#3 [100]', 'FinalExam [300]']]
#for feature in features:
#    dfr[feature] = scaler.fit_transform(dfr[feature])

#x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', 'Course Grade'], axis=1), dfr['Course Grade'])


# In[19]:


#fe = []
#for i in range(1, 13):
#    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
#    knn.fit(x_train, y_train)
#    pred = knn.predict(x_test)
#    fe.append(np.mean(cross_val_score(knn, x_test, y_test, cv=5)))


# In[20]:


#k_star = np.argmin(fe)
#print(k_star)
#knn = KNeighborsClassifier(n_neighbors = k_star, p = 2)
#knn.fit(x_train, y_train)
#pred = knn.predict(x_test)
#knn.score(x_test, y_test)*100

