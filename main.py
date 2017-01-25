# Code principal faisant tourner la routine

# Importation des librairies
import numpy as np
import scipy
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

# Importation des donnees
game_train = pd.read_csv("ML_TEST/game_teams_train.csv")
player_train = pd.read_csv("ML_TEST/game_player_teams_train.csv")

game_test = pd.read_csv("ML_TEST/game_teams_test.csv")
player_test = pd.read_csv("ML_TEST/game_player_teams_test.csv")

# On verifie que les donnees ont bien ete importees
print game_train.shape
print game_test.shape
print player_train.shape
print player_test.shape