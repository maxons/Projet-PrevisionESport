# Code principal faisant tourner la routine

# Importation des librairies
import numpy as np
import scipy
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import functions_data as fd
import functions_data as fd
from sklearn.ensemble import RandomForestClassifier

game_train_f = "ML_TEST/game_teams_train.csv"
player_train_f = "ML_TEST/game_player_teams_train.csv"

X_train, y_train = fd.prepareData(game_train_f, player_train_f)

# Graphique
color = y_train

plt.figure(1)
plt.subplot(311)
plt.scatter(X_train[:,0],X_train[:,1], s = 5,
         cmap = "jet", c=color)
plt.title("ACP")

plt.subplot(312)
plt.scatter(X_train[:,1],X_train[:,2], s = 5,
         cmap = "jet", c=color)
plt.title("ACP")

plt.subplot(313)
plt.scatter(X_train[:,2],X_train[:,3], s = 5,
         cmap = "jet", c=color)
plt.title("ACP")

plt.show()


# Random forest
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
	max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True)

# Apprentissage

forest = forest.fit(X_train,y_train)
print(1-forest.oob_score_)

# Erreur de prévision sur le test
1-forest.score(X_test,y_test)








































