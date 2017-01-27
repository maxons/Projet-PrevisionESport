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

pd.crosstab(index = nan_game[0], columns = "nbLines")
# 9 variables impliquées ou rien
pd.crosstab(index = nan_player[0], columns = "nbLines")
# 1-2-3-39 ou 40 variables

plt.figure(1)
plt.subplot(211)
plt.hist(nan_game)
plt.subplot(212)
plt.hist(nan_player)
plt.show()

# On veut savoir pour chacun des cas quelles sont les variables qui sont touchees
# Donnees des matches
nb_nan = (game.apply(np.isnan)*1).apply(sum,0)

# Donnees des joueurs
nb_nan = (player.apply(np.isnan)*1).apply(sum,1)

nan_1 = np.where(nb_nan == 1)[0]
col_nan_1 = (player.loc[nan_1].apply(np.isnan)*1).apply(sum, 0)


nan_2 = np.where(nb_nan == 2)[0]
col_nan_2 = (player.loc[nan_2].apply(np.isnan)*1).apply(sum, 0)


nan_3 = np.where(nb_nan == 3)[0]
col_nan_3 = (player.loc[nan_3].apply(np.isnan)*1).apply(sum, 0)


nan_39 = np.where(nb_nan == 39)[0]
col_nan_39 = (player.loc[nan_39].apply(np.isnan)*1).apply(sum, 0)


nan_40 = np.where(nb_nan == 40)[0]
col_nan_40 = (player.loc[nan_40].apply(np.isnan)*1).apply(sum, 0)




col_nan_1[np.where(col_nan_1 > 0)[0]]
col_nan_2[np.where(col_nan_2 > 0)[0]]
col_nan_3[np.where(col_nan_3 > 0)[0]]
col_nan_39[np.where(col_nan_39 > 0)[0]]
col_nan_40[np.where(col_nan_40 > 0)[0]]



























































