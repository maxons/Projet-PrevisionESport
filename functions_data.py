import pandas as pd

#------------
# Fonctions qui permettent de mettre en place les données
#------------

# Affiche l'ensemble des données contenues dans x
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# Fonction pour le faire dans les jeux de donnees type match
def replaceTFgame (data):
	colGame = ['first_blood', 'first_tower', 'first_inhibitor', 'first_baron', 'first_dragon']
	for col in colGame:
		data[col].replace(['t','f'], [1,0], inplace = True)
	return(data)

# Idem dans le jeu de donnees type players
def replaceTFplayer (data):
	colPlayer = ['first_blood_kill', 'first_blood_assist', 'first_tower_kill', 'first_tower_assist', 'first_inhibitor_kill', 'first_inhibitor_assist']
	for col in colPlayer:
		data[col].replace(['t','f'], [1,0], inplace = True)
	return(data)

# Nombre de nan par individu
def nan_per_ind (data):
	import pandas as pd
	import numpy as np
	nb_nan = (data.apply(np.isnan)*1).apply(sum,1)
	res = pd.DataFrame(nb_nan[np.where(nb_nan > 0)[0]])
	return res

# Precisions dans "nan_study.py"
def impute_player (player):
	import pandas as pd
	import numpy as np
	import sklearn as sk
	from sklearn.preprocessing import Imputer
	from sklearn.neighbors import NearestNeighbors
	from scipy import stats
	nan_player = nan_per_ind(player)
	nb_nan = (player.apply(np.isnan)*1).apply(sum,1)
	nan_2 = np.where(nb_nan == 2)[0]
	col_nan_2 = (player.loc[nan_2].apply(np.isnan)*1).apply(sum, 0)
	nan_3 = np.where(nb_nan == 3)[0]
	col_nan_3 = (player.loc[nan_3].apply(np.isnan)*1).apply(sum, 0)
	player['champion_id'] = player['champion_id'].astype('category')
	X = player.drop(['game_id', 'team_id', 'player_id'], 1)
	colToRem = col_nan_2[np.where(col_nan_2 > 0)[0]].index
	X_temp = X.drop(colToRem, 1)
	to_rem = np.where(nb_nan > 2)[0]
	X_2 = X_temp.drop(X_temp.index[to_rem], 0)
	nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X_2)
	distances, indices = nbrs.kneighbors(X_2)
	col_with_info = X[colToRem]
	col_with_info = col_with_info.drop(col_with_info.index[to_rem], 0)
	to_see = np.where((col_with_info.apply(np.isnan)*1).apply(sum, 1) > 0)[0]
	for ind in indices[to_see]:
		t_ind = col_with_info.index[ind]
		vals = col_with_info.loc[t_ind].apply(stats.mode,0)
		player = player.set_value(t_ind[0], colToRem[0], vals[0][0][0])
		player = player.set_value(t_ind[0], colToRem[1], vals[1][0][0])
	X = player.drop(['game_id', 'team_id', 'player_id'], 1)
	colToRem = col_nan_3[np.where(col_nan_3 > 0)[0]].index
	X_temp = X.drop(colToRem, 1)
	to_rem = np.where(nb_nan > 3)[0]
	X_3 = X_temp.drop(X_temp.index[to_rem], 0)
	X_3 = sk.preprocessing.normalize(X_3)
	nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(X_3)
	distances, indices = nbrs.kneighbors(X_3)
	col_with_info = X[colToRem]
	col_with_info = col_with_info.drop(col_with_info.index[to_rem], 0)
	to_see = np.where((col_with_info.apply(np.isnan)*1).apply(sum, 1) > 0)[0]
	for ind in indices[to_see]:
		t_ind = col_with_info.index[ind]
		vals = col_with_info.loc[t_ind].apply(np.mean,0)
		player = player.set_value(t_ind[0], colToRem[0], vals[0])
		player = player.set_value(t_ind[0], colToRem[1], vals[1])
		player = player.set_value(t_ind[0], colToRem[2], vals[2])
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	player[colToRem] = imp.fit_transform(player[colToRem])
	return(player)

def prepareData (file_game, file_players):
	import pandas as pd
	import numpy as np
	import sklearn as sk
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import Imputer
	game = pd.read_csv(file_game)
	player = pd.read_csv(file_players)
	# On veut remplacer tous les T/F par 1/0 dans les deux dataset
	game = replaceTFgame(game)
	player = replaceTFplayer(player)
	player = impute_player(player)
	# Elements que l'on veut jeter - variable choisie au hasard
	bool =  pd.isnull(game['first_blood'])
	# On garde en memoire les game_id pour les retirer aussi de l'autre jeu
	game_to_remove = np.unique(game['game_id'][bool])
	# On ne garde que les game_id juges valable
	game = game[:][~bool]
	# Maintenant on veut retirer ces matchs du jeu de donnees avec les joueurs
	# On va commencer par trier les donnees par game_id - team_id - player_id
	player.sort_values(by = ['game_id', 'team_id', 'player_id'], inplace = True)
	# Indices des matchs a retirer
	bool = pd.Series(player['game_id']).isin(game_to_remove)
	player = player[:][~bool]
	# On retourne sur game.
	# On veut remplacer créer une nouvelle variable valant 1 si l'équipe a gg, 0 sinon
	victory = (game['winner_id'] == game['team_id'])*1
	game['victory'] = victory
	# Grace a notre façon de trier, on a juste a faire la somme de cinq elements a la suite pour avoir la somme par equipe
	player_temp = player.drop(['game_id', 'team_id', 'player_id', 'champion_id','first_blood_kill',
	 'first_blood_assist', 'first_tower_kill', 'first_tower_assist', 'first_inhibitor_kill', 'first_inhibitor_assist'], 1)
	# On retire les valeurs manquantes de 'killing_sprees' et 'double_kills' en les estimant par la moyenne
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	player_temp = imp.fit_transform(player_temp)
	n = int(player_temp.shape[0]/5)
	m = player_temp.shape[1]
	# Matrice qui va contenir toutes les sommes
	res = np.arange(n*m).reshape(n, m)
	# On calcule les sommes
	for ii in range(0, n):
		res[ii:(ii+1)] = np.array(player_temp[5*ii:5*(ii+1)].sum(0)).reshape(1,m)
	res = pd.DataFrame(res)
	# On range la matrice obtenu dans game
	n = game.shape[0]
	game.index = (range(0, n))
	game = game.join(res)
	# ACP
	X = game.drop(['game_id', 'winner_id', 'team_id', 'victory'], 1)
	y = game['victory']
	return X, y




def prepare_few_variable (game_file, player_file):
	import pandas as pd
	import numpy as np
	game = pd.read_csv(game_file)
	game = replaceTFgame(game)
	player = pd.read_csv(player_file)
	player = replaceTFplayer(player)
	# Donnees des matches
	nb_nan = (game.apply(np.isnan)*1).apply(sum,0)
	game = game.drop(game.columns[nb_nan > 0], 1)
	# Qui a gagne?
	victory = (game['winner_id'] == game['team_id'])*1
	game['victory'] = victory
	# Donnees des joueurs
	# Individu par individu, le nombre de valeurs manquantes
	nb_nan = (player.apply(np.isnan)*1).apply(sum,1)
	# On regarde pour chacune des classes obtenues les variables impliquees
	# Indices de(s) individu(s) ou on a x valeurs manquantes
	nan_40 = np.where(nb_nan == 40)[0]
	col_nan_40 = (player.loc[nan_40].apply(np.isnan)*1).apply(sum, 0)
	# On ne garde que les variables associes a 40 valeurs manquantes
	col_to_drop = col_nan_40[np.where(col_nan_40 > 0)[0]].index
	player = player.drop(col_to_drop, 1)
	player.sort_values(by = ['game_id', 'team_id', 'gold_earned'], inplace = True)
	X = player.drop(['game_id', 'team_id', 'player_id'], 1)
	n = int(X.shape[0]/5)
	m = X.shape[1]
	M = m*5
	# Pour ces donnees, on ne va pas faire la somme des variables restantes comme on a pu le faire avant
	# On a trier les joueurs par or gagne, en considerant que l'or gagne dans une partie est revelateur
	# de l'importance qu'a eu le joueur dans la partie
	# On trie donc les joueurs par or gagne, et ensuite on va mettre les joueurs a la ligne pour creer de 
	# nouvelles variables
	# Matrice qui va contenir toutes les variables
	res = np.arange(n*M).reshape(n, M)
	# On calcule les sommes
	for ii in range(0, n):
		res[ii][0:m] = X.loc[ii]
		res[ii][m:2*m] = X.loc[ii+1] 
		res[ii][2*m:3*m] = X.loc[ii+2]
		res[ii][3*m:4*m] = X.loc[ii+3]
		res[ii][4*m:5*m] = X.loc[ii+4]
	res = pd.DataFrame(res)
	# On range la matrice obtenu dans game
	n = game.shape[0]
	game.index = (range(0, n))
	game = game.join(res)
	X = game.drop(['game_id', 'winner_id', 'team_id', 'victory'], 1)
	y = game['victory']
	return X, y


