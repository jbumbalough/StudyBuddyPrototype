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