# Code principal faisant tourner la routine

# Importation des librairies
import numpy as np
import scipy
import sklearn as sk
import csv


# Importation des donnees

file=open("ML_TEST/game_player_teams_train.csv","r")
player_train=csv.reader(file)
file=open("ML_TEST/game_teams_train.csv","r")
game_train=csv.reader(file)

file=open("ML_TEST/game_teams_test.csv","r")
game_test=csv.reader(file)
file=open("ML_TEST/game_teams_test.csv","r")
game_test=csv.reader(file)

# On verifie que l'importation c'est bien passee
