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

#------------
# Importation des donnees
#------------

game_train_f = "ML_TEST/game_teams_train.csv"
player_train_f = "ML_TEST/game_player_teams_train.csv"
X_train, y_train = fd.prepareData(game_train_f, player_train_f)

game_test_f = "ML_TEST/game_teams_test.csv"
player_test_f = "ML_TEST/game_player_teams_test.csv"
X_test, y_test = fd.prepareData(game_test_f, player_test_f)

#------------
# Apprentissage sur les donnees sans ACP
#------------

# Random forest
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
	max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True)

forest = forest.fit(X_train,y_train)
print(1-forest.oob_score_)

# Erreur de prévision sur le test
1-forest.score(X_test,y_test)
# 0.0346

#------------
# ACP
#------------

# On veut décider du bon nombre de composantes a garder
X_train_pca = sk.preprocessing.scale(X_train)
X_test_pca = sk.preprocessing.scale(X_test)

pca = PCA()
X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)
plt.plot(pca.explained_variance_ratio_)
plt.show()

# On en choisit 3 (explication de 90% de la variance)

pca = PCA(n_components = 3)
X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)
plt.plot(pca.explained_variance_ratio_)
plt.show()

# Graphique
color = y_train

plt.figure(1)
plt.subplot(311)
plt.scatter(X_train_pca[:,0],X_train_pca[:,1], s = 5,
         cmap = "jet", c=color)
plt.title("ACP")

plt.subplot(312)
plt.scatter(X_train_pca[:,1],X_train_pca[:,2], s = 5,
         cmap = "jet", c=color)
plt.title("ACP")
plt.show()


X_test_pca = pca.fit(X_test_pca).transform(X_test_pca)


#------------
# Apprentissage sur les ACP
#------------


forest = forest.fit(X_train_pca,y_train)
print(1-forest.oob_score_)

# Erreur de prévision sur le test
1-forest.score(X_test_pca,y_test)
# 0.0875

# On obtient une meilleure prediction sans faire de ACP
# Prendre plus de composantes?





































