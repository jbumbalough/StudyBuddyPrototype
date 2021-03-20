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
import sys
import scipy
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
np.set_printoptions(threshold=sys.maxsize)

##Getting the data
df = pd.read_csv("CourseBuddyDATA_SAMPLE1.csv")

##Transforming the data
dfr = df.replace('A', 5)
dfr = dfr.replace('B', 4)
dfr = dfr.replace('C', 3)
dfr = dfr.replace('D', 2)
dfr = dfr.replace('F', 1)

#Test student data
a_original = pd.DataFrame({'S#' : [100], 'Prerequisite 1': [5], 'Prerequisite 2': [5], 'Prerequisite 3': [5], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
b_original = pd.DataFrame({'S#' : [101], 'Prerequisite 1': [4], 'Prerequisite 2': [4], 'Prerequisite 3': [4], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
c_original = pd.DataFrame({'S#' : [103], 'Prerequisite 1': [3], 'Prerequisite 2': [3], 'Prerequisite 3': [3], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
d_original = pd.DataFrame({'S#' : [104], 'Prerequisite 1': [2], 'Prerequisite 2': [2], 'Prerequisite 3': [2], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
f_original = pd.DataFrame({'S#' : [105], 'Prerequisite 1': [1], 'Prerequisite 2': [1], 'Prerequisite 3': [1], 'Class Participation [100]' : [0], 'HW#1 [100]' : [0], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})

a_final = pd.DataFrame({'S#' : [106], 'Prerequisite 1': [5], 'Prerequisite 2': [5], 'Prerequisite 3': [5], 'Class Participation [100]' : [90], 'HW#1 [100]' : [90], 'HW#2 [100]' : [90], 'HW#3 [100]' : [90], 'Test#1 [100]' : [90], 'Test#2 [100]' : [90], 'Test#3 [100]' : [90], 'FinalExam [300]' : [270], 'Course Grade' : [5]})
b_final = pd.DataFrame({'S#' : [107], 'Prerequisite 1': [4], 'Prerequisite 2': [4], 'Prerequisite 3': [4], 'Class Participation [100]' : [80], 'HW#1 [100]' : [80], 'HW#2 [100]' : [80], 'HW#3 [100]' : [80], 'Test#1 [100]' : [80], 'Test#2 [100]' : [80], 'Test#3 [100]' : [80], 'FinalExam [300]' : [240], 'Course Grade' : [4]})
c_final = pd.DataFrame({'S#' : [108], 'Prerequisite 1': [3], 'Prerequisite 2': [3], 'Prerequisite 3': [3], 'Class Participation [100]' : [70], 'HW#1 [100]' : [70], 'HW#2 [100]' : [70], 'HW#3 [100]' : [70], 'Test#1 [100]' : [70], 'Test#2 [100]' : [70], 'Test#3 [100]' : [70], 'FinalExam [300]' : [210], 'Course Grade' : [3]})
d_final = pd.DataFrame({'S#' : [109], 'Prerequisite 1': [2], 'Prerequisite 2': [2], 'Prerequisite 3': [2], 'Class Participation [100]' : [60], 'HW#1 [100]' : [60], 'HW#2 [100]' : [60], 'HW#3 [100]' : [60], 'Test#1 [100]' : [60], 'Test#2 [100]' : [60], 'Test#3 [100]' : [60], 'FinalExam [300]' : [180], 'Course Grade' : [2]})
f_final = pd.DataFrame({'S#' : [110], 'Prerequisite 1': [1], 'Prerequisite 2': [1], 'Prerequisite 3': [1], 'Class Participation [100]' : [50], 'HW#1 [100]' : [50], 'HW#2 [100]' : [50], 'HW#3 [100]' : [50], 'Test#1 [100]' : [50], 'Test#2 [100]' : [50], 'Test#3 [100]' : [50], 'FinalExam [300]' : [150], 'Course Grade' : [1]})

dfn = a_final.append(b_final, ignore_index = True).append(c_final, ignore_index = True).append(d_final, ignore_index = True).append(f_final, ignore_index = True)


# In[2]:


##Predicting final grade, based on all other grades for all students
def predict(original):
    copy = original.copy()
    features = ['Class Participation [100]', 'HW#1 [100]', 'Test#1 [100]', 'HW#2 [100]', 'Test#2 [100]', 'HW#3 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']
    #features = copy.columns
    for i in features:
        print("Feature: ", i)
        x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', i], axis=1), dfr[i])
        x_test = copy.drop(['S#', i], axis=1)
        nb = GaussianNB().fit(x_train, y_train)
        pred = nb.predict(x_test)
        print("Prediction: ", pred)
        copy[i] = pred
        print("\n===============\n")


# In[3]:


#Recommending similar students
def similarStudent(stud):
    for s in range(dfr.size):
        d = scipy.spatial.distance.cdist(stud, dfr, metric='euclidean')
        #print(d)
        m = min(i for i in d[s] if i > 0)
        #print(m)
        a = np.where(d == m)
        print("Student#: ", a[1][0])
        #return a[1][0]


# In[4]:


##Testing grade items that consist of different point values, multiple questions or topics
class GradeItem:
    def __init__(self, n, p, q, t):
        self.name = n
        self.points = p
        self.questions = q
        self.topics = t
        self.score = 0
        
    def display(self):
        print(self.name, ": ", self.score, "/", self.points)
        print(self.topics)
        print(self.questions)


# In[5]:


predict(a_original)


# In[6]:


similarStudent(a_original)


# In[7]:


similarStudent(a_final)


# In[ ]:




