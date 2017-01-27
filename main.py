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
from sklearn.ensemble import RandomForestClassifier

#------------
# Importation des donnees
#------------

game_train_f = "ML_TEST/game_teams_train.csv"
player_train_f = "ML_TEST/game_player_teams_train.csv"
print("On importe les donnees d'apprentissage")
X_train, y_train = fd.prepareData(game_train_f, player_train_f)

game_test_f = "ML_TEST/game_teams_test.csv"
player_test_f = "ML_TEST/game_player_teams_test.csv"
print("On importe les donnees de test")
X_test, y_test = fd.prepareData(game_test_f, player_test_f)

#------------
# Apprentissage sur les donnees sans ACP
#------------

# Random forest
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
	max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True)

print("Random forest sur les donnees brutes")
forest = forest.fit(X_train,y_train)
print("Score de bagging")
print(1-forest.oob_score_)

print("Erreur de prevision sur le test")
print(1-forest.score(X_test,y_test))
# 0.0346

#------------
# ACP
#------------

print("On procede a un ACP pour resumer l'information")
# On veut decider du bon nombre de composantes a garder
X_train_pca = sk.preprocessing.scale(X_train)
X_test_pca = sk.preprocessing.scale(X_test)

pca = PCA()
X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)
plt.plot(pca.explained_variance_ratio_)
plt.title("Ratio variance expliquee en fonction du nombre de composante")
plt.show()

# On en choisit 2 (explication de 90% de la variance)

pca = PCA(n_components = 3)
X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)

# Graphique
color = y_train

plt.figure(1)
plt.subplot(211)
plt.scatter(X_train_pca[:,0],X_train_pca[:,1], s = 5,
         cmap = "jet", c=color)
plt.title("1ere composante en fonction de la 2eme")

plt.subplot(212)
plt.scatter(X_train_pca[:,1],X_train_pca[:,2], s = 5,
         cmap = "jet", c=color)
plt.title("2eme composante en fonction de la 3eme")
plt.show()

# On remarque une belle separation entre les deux groupes


X_test_pca = pca.fit(X_test_pca).transform(X_test_pca)


#------------
# Apprentissage sur les ACP
#------------

print("Random forest sur l'ACP")
forest = forest.fit(X_train_pca,y_train)
print("Score de baggig")
print(1-forest.oob_score_)

print("Erreur de prevision sur le test")
print(1-forest.score(X_test_pca,y_test))
# 0.0824

# On obtient une meilleure prediction sans faire de ACP
# Prendre plus de composantes?

print("On teste d'autres nombre de composantes")
n_compo = range(3,21)
res = np.zeros(18)

for ii in range(0,18):
	X_train_pca = sk.preprocessing.scale(X_train)
	X_test_pca = sk.preprocessing.scale(X_test)
	pca = PCA(n_components = n_compo[ii])
	X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)
	X_test_pca = pca.fit(X_test_pca).transform(X_test_pca)
	forest = forest.fit(X_train_pca,y_train)
	res[ii] = 1-forest.score(X_test_pca,y_test)

plt.plot(res)
plt.title("Erreur de test en fonction du nombre de composantes")
plt.show()

# On voit que le mieux est obtenu pour 2 composantes
# On admet que la meilleure solution est obtenue sans ACP. On va chercher a optimiser l'algorithme

print("Optimisation du random forest")
from sklearn.model_selection import GridSearchCV
param = [{"max_features":list(range(2,15))}]
best_feat = GridSearchCV(RandomForestClassifier(n_estimators=100),param,cv=5,n_jobs=-1)
best_feat = best_feat.fit(X_train, y_train)
# parametre optimal
best_feat.best_params_

forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, 
	min_samples_split=2, min_samples_leaf=1, max_features=6, max_leaf_nodes=None, bootstrap=True, oob_score=True)

# Apprentissage
forest = forest.fit(X_train,y_train)
print("Nouveau score de bagging")
print(1-forest.oob_score_)
# erreur de prevision sur le test
print("Nouvelle erreur de test")
print(1-forest.score(X_test,y_test))
#0.0356

# Prevision
y_chap = forest.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_test, y_chap)
print(table)


#------------
# Cas ou on prend peu de donnees
#------------

print("On veut maintenant quand on a peu de variables")

game_train_f = "ML_TEST/game_teams_train.csv"
player_train_f = "ML_TEST/game_player_teams_train.csv"
X_train, y_train = fd.prepare_few_variable(game_train_f, player_train_f)

game_test_f = "ML_TEST/game_teams_test.csv"
player_test_f = "ML_TEST/game_player_teams_test.csv"
X_test, y_test = fd.prepare_few_variable(game_test_f, player_test_f)


print("random forest")
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
	max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True)

forest = forest.fit(X_train,y_train)
print("Score de bagging")
print(1-forest.oob_score_)

# Erreur de prevision sur le test
print("Erreur de prediction")
print(1-forest.score(X_test,y_test))
# 0.5

#------------
# ACP
#------------

print("On veut essayer d'ameliorer avec un ACP")

# On veut decider du bon nombre de composantes a garder
X_train_pca = sk.preprocessing.normalize(X_train)
X_test_pca = sk.preprocessing.normalize(X_test)

pca = PCA()
X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)
# plt.plot(pca.explained_variance_ratio_)
# plt.show()

# On en choisit 2 

pca = PCA(n_components = 3)
X_train_pca = pca.fit(X_train_pca).transform(X_train_pca)
# plt.plot(pca.explained_variance_ratio_)
# plt.show()

# Graphique
color = y_train

plt.figure(1)
plt.subplot(211)
plt.scatter(X_train_pca[:,0],X_train_pca[:,1], s = 5,
         cmap = "jet", c=color)
plt.title("1ere composante en fonction de la 2eme")

plt.subplot(212)
plt.scatter(X_train_pca[:,1],X_train_pca[:,2], s = 5,
         cmap = "jet", c=color)
plt.title("2eme composante en fonction de la 3eme")
plt.show()

X_test_pca = pca.fit(X_test_pca).transform(X_test_pca)


#------------
# Apprentissage sur les ACP
#------------


forest = forest.fit(X_train_pca,y_train)
print("Score de bagging")
print(1-forest.oob_score_)

# Erreur de prevision sur le test
print("Erreur de prediction sur le test")
print(1-forest.score(X_test_pca,y_test))
# 0.51

# Pas bon du tout, mais on pouvait s'en douter avec les plot de ACP.








