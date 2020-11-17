###
# Study Buddy
# MovieRecommender
# A basic prototype for a movie reccomender system using surprise.
###

import numpy as np
import surprise as surprise
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movie dataset
data = Dataset.load_builtin('ml-100k')

# Using SVD algorithm
algo = SVD()

# 5 fold cross validation
cross_validate(algo, data, measures = ['RMSE', 'MAE'], cv=5, verbose=True)

