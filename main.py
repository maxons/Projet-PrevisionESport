# Code principal faisant tourner la routine

# Importation des librairies
import numpy as np
import scipy
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import math

# Importation des donnees
game_train = pd.read_csv("ML_TEST/game_teams_train.csv")
player_train = pd.read_csv("ML_TEST/game_player_teams_train.csv")

game_test = pd.read_csv("ML_TEST/game_teams_test.csv")
player_test = pd.read_csv("ML_TEST/game_player_teams_test.csv")

# On verifie que les donnees ont bien ete importees
print(game_train.shape)
print(game_test.shape)
print(player_train.shape)
print(player_test.shape)

# Mis en forme des donnees
# - f/t -> 0/1, on garde les NaN
colGame = ['first_blood', 'first_tower', 'first_inhibitor', 'first_baron', 'first_dragon']
colPlayer = ['first_blood_kill', 'first_blood_assist', 'first_tower_kill', 'first_tower_assist', 'first_inhibitor_kill', 'first_inhibitor_assist']

game_train[colGame]
player_train[colPlayer]

temp = game_train[colGame]
ind = np.where(temp == 'f')

#--------------
# On souhaite faire les predictions uniquement avec les donnees propres aux matchs (pour le moment)
#--------------

