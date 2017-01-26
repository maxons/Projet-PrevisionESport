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


