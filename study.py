###
# Study Buddy
# Coursebuddy Prototype
# A basic prototype for the Coursebuddy system using surprise.
###

import numpy as np
import pandas as pd
import surprise as surprise
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

df = pd.read_csv("Prototype Data - Sheet1.csv")
print(df)

algo = SVD()
rd = {"Course" : ["CS 100", "CS 101", "CS 102", "CS 103"],
      "User" : ["Bob", "Sally", "Tim", "Samantha"],
      "FinalGrade" : [99, 88, 77, 66]}
df = pd.DataFrame(rd)
reader = Reader(rating_scale = (0, 100))
data = Dataset.load_from_df(df[["User", "Course", "FinalGrade"]], reader)
cross_validate(algo, data, measures = ['RMSE', 'MAE'], cv=4, verbose=True)

from collections import defaultdict
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
	
	trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
preds = algo.test(testset)
top = get_top_n(preds, n=10)
print(top)

for key, value in top.items():
    print(key)
    for i in range(len(value)):
        print(value[i])
		
		