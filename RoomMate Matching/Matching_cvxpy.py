import numpy as np
from cvxpy import *
import itertools
import pandas as pd

import csv
import json

  
#--- code to get the hamming distance--
# filename="preference"
# book= pd.ExcelFile(r'C:\Users\sowmy\Desktop\Cvxopt\Sampletwitter.xlsx')
# df= book.parse("Sheet4")

# #Base data
# #users = df["Name"]
# print (users)


# from sklearn.metrics.pairwise import pairwise_distances
# print(1 - pairwise_distances(df.T, metric = "hamming"))

# f = open('C:/Users/sowmy/Downloads/'+ filename +'.csv','w',encoding="utf-8",newline='\n') # write mode binary
# fw = csv.writer(f) # create csv writer
# fw.writerow(['index','name', 'screen_name', 'created_at', 'hashtags_count',
#                       'hashtag_text', 'text=tweet.fulltext']) 
# fw.writerows(1 - pairwise_distances(df.T, metric = "hamming"))
# f.close() # close file

#--- code to perform linear matching--

N=50

book= pd.ExcelFile(r'C:\Users\sowmy\Desktop\Cvxopt\preferencefile.xlsx')
distances = book.parse("Sheet3")
distances.to_numpy()

#preference file is the file that has the hamming distance

# Binary Dec.VARS
x = Variable((49, 50), boolean=True)
#u = Variable(50, integer=True)

# CONSTRAINTS
constraints = []
for j in range(49):
    indices = np.hstack((np.arange(0, j), np.arange(j + 1, 49)))
    constraints.append(sum(x[indices, j]) == 1)
for i in range(49):
    indices = np.hstack((np.arange(0, i), np.arange(i + 1, 49)))
    constraints.append(sum(x[i, indices]) == 1)

# for i in range(1, N):
#     for j in range(1, N):
#         if i != j:
#             constraints.append(u[i] - u[j] + N*x[i, j] <= N-1)

# OBJ
obj = Minimize(sum(multiply(distances, x)))

# SOLVE
prob = Problem(obj, constraints)
prob.solve(verbose=True)
print(prob.value)
print(x.value)

file="output"

f = open('C:/Users/sowmy/Downloads/'+ file +'.csv','w',encoding="utf-8",newline='\n') # write mode binary
fw = csv.writer(f) # create csv writer
fw.writerow(['v1','v2', 'v3', 'v4', 'v5','v6', 'v7']) 
fw.writerows(x.value)
f.close()
