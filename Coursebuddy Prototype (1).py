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
#dfn.append(b_final, ignore_index = True)
#dfn.append(c_final, ignore_index = True)
#dfn.append(d_final, ignore_index = True)
#dfn.append(f_final, ignore_index = True)
dfn


# In[4]:


##Predicting final grade, based on all other grades for all students
x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', 'Course Grade'], axis=1), dfr['Course Grade'])

##Finding the best k neighbors
fe = []
for i in range(1, 13):
    knn = KNeighborsClassifier(n_neighbors = i, p = 2)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    fe.append(np.mean(cross_val_score(knn, x_test, y_test, cv=5)))
    
##The Prediction
k_star = np.argmin(fe)
print(k_star)
knn = KNeighborsClassifier(n_neighbors = k_star, p = 2)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(pred)
knn.score(x_test, y_test)*100


# In[5]:


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


# In[6]:


##Method for calculating accuracy
def acc(obs, accept):
    return abs(((obs - accept) / accept) * 100)


# In[7]:


##Method for predicting each grade item for a student based on previous grade items. This uses these predictions as data for future items.
def predict(original):
    copy = original.copy()
    features = ['Class Participation [100]', 'HW#1 [100]', 'Test#1 [100]', 'HW#2 [100]', 'Test#2 [100]', 'HW#3 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']
    #features = copy.columns
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


# In[8]:


##A test for a generic A grade student with no grades, only prereqs
predict(a_original)


# In[9]:


##A test for a generic B grade student with no grades, only prereqs
predict(b_original)


# In[10]:


#############################################################
##Testing grade items that consist of different point values, multiple questions or topics


# In[11]:


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


# In[12]:


h1 = GradeItem("HW 1", 100, ["q1", "q2", "q3"], ["t1"])
h1.display()
h1.score = 88
h1.display()


# In[13]:


##Due to limitations of dataframes, I do not believe it possible to nest the GradeItem class into the dataframe
new_stud = pd.DataFrame({'S#' : [102], 'Prerequisite 1': [5], 'Prerequisite 2': [4], 'Prerequisite 3': [3], 'Class Participation [100]' : [0], GradeItem("HW#1", 100, ["q1", "q2", "q3"], ["t1"]).name : [88], 'HW#2 [100]' : [0], 'HW#3 [100]' : [0], 'Test#1 [100]' : [0], 'Test#2 [100]' : [0], 'Test#3 [100]' : [0], 'FinalExam [300]' : [0], 'Course Grade' : [0]})
new_stud


# In[14]:



for i in new_stud.columns:
    print("Feature: ", i)
    print(new_stud[i], "\n")
    #for j in new_stud[i]:
        #print(j.name, " ", j.points, " ", j.questions, " ", j.topics)


# In[15]:


#Recommending similar students
x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#'], axis=1), dfr['S#'])
knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(y_test)
print(pred)
print(knn.score(x_test, y_test)*100)


# In[16]:


###Testing accuracy by giving a set of initial and final scores

##Predicting each of the grade items over time, based on available previous data
a_copy = a_original.copy()
pe = []

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
        pe.append(acc(pred, j))
        print("PE: ", acc(pred, j))
    a_copy[i] = pred
    print("\n===============\n")
    
print(a_original)
print("\n===============\n")
print(a_copy)
print("\n===============\n")
print(a_final)

av = sum(pe) / len(pe)
print("Average PE: ", av)
print("Overall Accuracy: ", 100-av)


# In[17]:


############Linear Regression (accuracy plummets and bounds are not upheld)


# In[18]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(x_train, y_train)
print(lr.score(x_train, y_train))
print('intercept:', lr.intercept_)
print('slope:', lr.coef_)


# In[19]:


result = lr.predict(x_test)
print(result)
print(y_test)


# In[20]:


def predict(original):
    copy = original.copy()
    features = ['Class Participation [100]', 'HW#1 [100]', 'Test#1 [100]', 'HW#2 [100]', 'Test#2 [100]', 'HW#3 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']
    #features = copy.columns
    for i in features:
        print("Feature: ", i)
        x_train, x_test, y_train, y_test = train_test_split(dfr.drop(['S#', i], axis=1), dfr[i])
        x_test = copy.drop(['S#', i], axis=1)
        lr = LinearRegression().fit(x_train, y_train)
        pred = lr.predict(x_test)
        print("Prediction: ", pred)
        copy[i] = pred
        print("\n===============\n")


# In[21]:


predict(a_original)


# In[22]:


predict(b_original)


# In[23]:


################Naive Bayes (much better accuracy, but fails with low grade students due to the data its been trained on)


# In[24]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB().fit(x_train, y_train)


# In[25]:


result = nb.predict(x_test)
print(x_test)
print(y_test)
print(result)


# In[26]:


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


# In[27]:


predict(a_original)


# In[28]:


predict(b_original)


# In[29]:


predict(c_original)


# In[30]:


predict(d_original)


# In[31]:


predict(f_original)


# In[32]:


###Testing with a few added values
dfrn = dfr.append(dfn, ignore_index = True)
dfrn


# In[33]:


def predictn(original):
    copy = original.copy()
    features = ['Class Participation [100]', 'HW#1 [100]', 'Test#1 [100]', 'HW#2 [100]', 'Test#2 [100]', 'HW#3 [100]', 'Test#3 [100]', 'FinalExam [300]', 'Course Grade']
    #features = copy.columns
    for i in features:
        print("Feature: ", i)
        x_train, x_test, y_train, y_test = train_test_split(dfrn.drop(['S#', i], axis=1), dfrn[i])
        x_test = copy.drop(['S#', i], axis=1)
        nb = GaussianNB().fit(x_train, y_train)
        pred = nb.predict(x_test)
        print("Prediction: ", pred)
        copy[i] = pred
        print("\n===============\n")
    print("\n===============\n")


# In[34]:


predictn(a_original)
predictn(b_original)
predictn(c_original)
predictn(d_original)
predictn(f_original)


# In[ ]:




