#!/usr/bin/env python
# coding: utf-8

# In[1]:


###
# Study Buddy
# MovieRecommender
# A basic prototype for a movie reccomender system using surprise.
###

import numpy as np
import pandas as pd
import surprise as surprise
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movie dataset
dataset = Dataset.load_builtin('ml-100k')

# Using SVD algorithm
algo = SVD()

# 5 fold cross validation
#cross_validate(algo, dataset, measures = ['RMSE', 'MAE'], cv=5, verbose=True)


# In[10]:


df = pd.DataFrame(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), columns=['user', 'movie', 'rating'])
df.head()
reader = surprise.Reader()
data = surprise.Dataset.load_from_df(df, reader)
algo.fit(data.build_full_trainset())
pred = algo.predict(uid = '1', iid = '1')
print(pred.est)

