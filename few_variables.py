# On va regarder a l'information que l'on peut tirer des donnees ou on a peu d'information


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



#------------
# Etude de la distribution des NaN
#------------


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

# On veut savoir pour chacun des cas quelles sont les variables qui sont touchees
# Donnees des matches
nb_nan = (game.apply(np.isnan)*1).apply(sum,0)

# Donnees des joueurs
# Individu par individu, le nombre de valeurs manquantes
nb_nan = (player.apply(np.isnan)*1).apply(sum,1)

# On regarde pour chacune des classes obtenues les variables impliquees
# Indices de(s) individu(s) ou on a x valeurs manquantes

nan_39 = np.where(nb_nan == 39)[0]
col_nan_39 = (player.loc[nan_39].apply(np.isnan)*1).apply(sum, 0)

nan_40 = np.where(nb_nan == 40)[0]
col_nan_40 = (player.loc[nan_40].apply(np.isnan)*1).apply(sum, 0)

# On regarde uniquement les variables impliquees pour x valeurs manquantes
col_nan_39[np.where(col_nan_39 == 0)[0]]
col_nan_40[np.where(col_nan_40 == 0)[0]]

# On ne garde que les variables associes a 40 valeurs manquantes
col_to_drop = col_nan_40[np.where(col_nan_40 > 0)[0]].index
player = player.drop(col_to_drop, 1)

# Idem dans l'autre dataset
col_to_drop = nb_nan[np.where(nb_nan > 0)[0]].index
game = game.drop(col_to_drop, 1)

# Qui a gagne?
victory = (game['winner_id'] == game['team_id'])*1
game['victory'] = victory

player.sort_values(by = ['game_id', 'team_id', 'gold_earned'], inplace = True)
X = player.drop(['game_id', 'team_id', 'player_id'], 1)

n = int(X.shape[0]/5)
m = X.shape[1]
M = m*5

# Matrice qui va contenir toutes les variables
res = np.arange(n*M).reshape(n, M)
# On calcule les sommes
for ii in range(0, n):
	res[ii][0:6] = X.loc[ii]
	res[ii][6:12] = X.loc[ii+1] 
	res[ii][12:18] = X.loc[ii+2]
	res[ii][18:24] = X.loc[ii+3]
	res[ii][24:30] = X.loc[ii+4]

res = pd.DataFrame(res)

# On range la matrice obtenu dans game
n = game.shape[0]

game.index = (range(0, n))
game = game.join(res)

X = game.drop(['game_id', 'winner_id', 'team_id', 'victory'], 1)
y = game['victory']




















































