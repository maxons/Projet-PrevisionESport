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
# Dans game_train: on remarque que pour certains matchs on a aucune information, on veut les supprimer
# Rem: soit on a tout, soit on a rien. On va chercher pour une variable seulement.

# Elements que l'on veut jeter
bool =  pd.isnull(game_train['first_blood'])
# On garde en memoire les game_id pour les retirer aussi de l'autre jeu
game_to_remove = np.unique(game_train['game_id'][bool])
# On ne garde que les game_id juges valable
game_train = game_train[:][~bool]

# On retire 271 matchs sur 1864, soit 15% des données. Cela fait beaucoup de pertes...

# Maintenant on veut retirer ces matchs du jeu de donnees avec les joueurs
# On va commencer par trier les donnees par game_id - team_id - player_id
player_train.sort_values(by = ['game_id', 'team_id', 'player_id'], inplace = True)
# Indices des matchs a retirer
bool = pd.Series(player_train['game_id']).isin(game_to_remove)
player_train = player_train[:][~bool]

