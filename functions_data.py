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

def prepareData (file_game, file_players):
	import pandas as pd
	import numpy as np
	import sklearn as sk
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import Imputer
	game = pd.read_csv(file_game)
	player = pd.read_csv(file_players)
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
	# On veut remplacer tous les T/F par 1/0 dans les deux dataset
	game = replaceTFgame(game)
	player = replaceTFplayer(player)
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





