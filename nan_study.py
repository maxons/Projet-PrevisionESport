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
# Individu par individu, le nombre de valeurs manquantes
nb_nan = (player.apply(np.isnan)*1).apply(sum,1)

# On regarde pour chacune des classes obtenues les variables impliquees
# Indices de(s) individu(s) ou on a x valeurs manquantes
nan_1 = np.where(nb_nan == 1)[0]
# Somme sur chaque variable pour savoir combien de fois la variable est impliquee pour x valeurs manquantes
col_nan_1 = (player.loc[nan_1].apply(np.isnan)*1).apply(sum, 0)

nan_2 = np.where(nb_nan == 2)[0]
col_nan_2 = (player.loc[nan_2].apply(np.isnan)*1).apply(sum, 0)

nan_3 = np.where(nb_nan == 3)[0]
col_nan_3 = (player.loc[nan_3].apply(np.isnan)*1).apply(sum, 0)

nan_39 = np.where(nb_nan == 39)[0]
col_nan_39 = (player.loc[nan_39].apply(np.isnan)*1).apply(sum, 0)

nan_40 = np.where(nb_nan == 40)[0]
col_nan_40 = (player.loc[nan_40].apply(np.isnan)*1).apply(sum, 0)

# On regarde uniquement les variables impliquees pour x valeurs manquantes
col_nan_1[np.where(col_nan_1 > 0)[0]]
col_nan_2[np.where(col_nan_2 > 0)[0]]
col_nan_3[np.where(col_nan_3 > 0)[0]]
col_nan_39[np.where(col_nan_39 > 0)[0]]
col_nan_40[np.where(col_nan_40 > 0)[0]]

# On remarque que dans chacun des classes, il manque toujours les meme variable
# Mais entre chaque classe on ne retrouve pas les meme variables, sauf dans les deux dernieres
# La difference entre 39 et 40 se fait pour le champion_id

# Le cas ou on a 1 valeur manquante ne nous interesse pas car il ne concerne que le player_id

# Dans les cas ou il nous manque 2 ou 3 variables, on va estimer les valeurs manquantes grace a du kNN
# On garde les individus ayant un nombre <= 3 valeurs manquantes

# On doit retirer les trois premieres colonnes

X_incomplete = player.drop(['game_id', 'team_id', 'player_id'], 1)

to_keep = np.where(nb_nan < 4)[0]

from fancyimpute import KNN
X = KNN(k=3).complete(X_incomplete.loc[to_keep])

# N'est pas satisfaisant car les estimations ne sont pas discretes. 





















































