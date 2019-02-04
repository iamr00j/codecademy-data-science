# Machine Learning is the science of getting computers to learn and act like humans do, 
# and improve their learning over time in autonomous fashion, by feeding them data and information 
# in the form of observations and real-world interactions. It has taken the Data Science world by storm.
# It can be branched out into the following categories:
#   Supervised Learning - when the data is labeled and the program learns to predict the output from the 
#                         input data. For instance, a supervised learning algorithm for credit card fraud 
#                         detection would take as input a set of recorded transactions. It would learn what 
#                         makes a transaction likely to be fraudulent. Then, for each new transaction, the 
#                         program would predict if it is fraudulent or not.
#   Unsupervised Learning - where the data is unlabeled and the program learns to recognize the inherent 
#                           structure of the input data. For the same fraud example, the model would take in 
#                           a bunch of transactions with no indication of if they are fraudulent or not, and 
#                           it would group them based on patterns it sees. It might discover two groups, 
#                           fraudulent and legitimate.

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

mu = 1
std = 0.5
mu2 = 4.188

np.random.seed(100)

xs = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.75,std,100)), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100))

ys = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100)), np.random.normal(0.75,std,100))

values = list(zip(xs, ys))

model = KMeans(init='random', n_clusters=2)

results = model.fit_predict(values)

plt.scatter(xs, ys, c=results, alpha=0.6)

colors = ['#6400e4', '#ffc740']

for i in range(2):
  points = np.array([values[j] for j in range(len(values)) if results[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.6)
  
plt.title('Codecademy Mobile Feedback - Data Science')

plt.xlabel('Learn Python')
plt.ylabel('Learn SQL')
  
plt.show()
