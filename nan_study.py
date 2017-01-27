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


#------------
# Approches pour estimation: fancyimpute
#------------

# On doit retirer les trois premieres colonnes
X = player.drop(['game_id', 'team_id', 'player_id'], 1)
X['champion_id'] = X['champion_id'].astype('category')

X['first_tower_kill'] = X['first_tower_kill'].astype('category')
X['first_tower_assist'] = X['first_tower_assist'].astype('category')
X['first_inhibitor_kill'] = X['first_inhibitor_kill'].astype('category')
X['first_inhibitor_assist'] = X['first_inhibitor_assist'].astype('category')
X['first_blood_assist'] = X['first_blood_assist'].astype('category')
X['first_blood_kill'] = X['first_blood_kill'].astype('category')

# Plus de 3 valeurs manquantes?
to_keep = np.where(nb_nan < 4)[0]
X = X.loc[to_keep]

from fancyimpute import KNN
X_complete = pd.DataFrame(KNN(k=3).complete(X))

# N'est pas satisfaisant car les estimations ne sont pas discretes. 

#------------
# Creation propre methode
#------------

# On va le faire "a la main"
from sklearn.neighbors import NearestNeighbors

X = player.drop(['game_id', 'team_id', 'player_id'], 1)
X['champion_id'] = X['champion_id'].astype('category')



# On se place dans le cas ou on a deux valeurs manquantes


# On retire les deux colonnes porteuses des valeurs
colToRem = col_nan_2[np.where(col_nan_2 > 0)[0]].index
X_temp = X.drop(colToRem, 1)
# On retire les individus ayant plus de deux valeurs manquantes, entre autre 3
to_rem = np.where(nb_nan > 2)[0]
X_2 = X_temp.drop(X_temp.index[to_rem], 0)

# On utilise 'ball_tree' car adaptee bcp variables
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X_2)
distances, indices = nbrs.kneighbors(X_2)
# indices renvoie les indices sur plus proches voisins
# mais il a remis les indices a 0, pas de memoire des vrais indices de X_2

# Variable qui contient les deux colonnes que l'on souhaite reconstruire
col_with_info = X[colToRem]
col_with_info = col_with_info.drop(col_with_info.index[to_rem], 0)

np.isnan

from scipy import stats


# Lignes qui nous interessent
to_see = np.where((col_with_info.apply(np.isnan)*1).apply(sum, 1) > 0)[0]
for ind in indices[to_see]:
	# Vrais indices
	t_ind = col_with_info.index[ind]
	# mode des valeurs a remplacer
	vals = col_with_info.loc[t_ind].apply(stats.mode,0)
	# On effectue la mise a jour
	player = player.set_value(t_ind[0], colToRem[0], vals[0][0][0])
	player = player.set_value(t_ind[0], colToRem[1], vals[1][0][0])

# On verifie que player n'a plus d'individus avec deux variables manquantes

nb_nan = (player.apply(np.isnan)*1).apply(sum,1)
nan_2 = np.where(nb_nan == 2)[0]
col_nan_2 = (player.loc[nan_2].apply(np.isnan)*1).apply(sum, 0)

# C'est bon!








































