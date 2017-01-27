# On veut etudier les NaN et creer mettre en place du kNN pour les retirer

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import functions_data as fd
import matplotlib.pyplot as plt


game_train_f = "ML_TEST/game_teams_train.csv"
game = pd.read_csv(game_train_f)
game = fd.replaceTFgame(game)

player_train_f = "ML_TEST/game_player_teams_train.csv"
player = pd.read_csv(player_train_f)
player = fd.replaceTFplayer(player)

def nan_per_ind (data):
	nb_nan = (data.apply(np.isnan)*1).apply(sum,1)
	res = pd.DataFrame(nb_nan[np.where(nb_nan > 0)[0]])
	return res

nan_game = nan_per_ind(game)
nan_player = nan_per_ind(player)

pd.crosstab(index = nan_game[0], columns = "lol")
pd.crosstab(index = nan_player[0], columns = "lol")



plt.figure(1)
plt.subplot(211)
plt.hist(nan_game)
plt.subplot(212)
plt.hist(nan_player)
plt.show()

