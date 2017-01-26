# Code principal faisant tourner la routine

# Importation des librairies
import numpy as np
import scipy
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# Importation des donnees
game_train = pd.read_csv("ML_TEST/game_teams_train.csv")
player_train = pd.read_csv("ML_TEST/game_player_teams_train.csv")

game_test = pd.read_csv("ML_TEST/game_teams_test.csv")
player_test = pd.read_csv("ML_TEST/game_player_teams_test.csv")

# On verifie que les donnees ont bien ete importees
print(game_train.shape)
print(game_test.shape)
print(player_train.shape)
print(player_test.shape)

#------------
# Rapide mise en forme des donnees
#------------

# Dans game_train: on remarque que pour certains matchs on a aucune information, on veut les supprimer
# Rem: soit on a tout, soit on a rien. On va chercher pour une variable seulement.

# Elements que l'on veut jeter
bool =  pd.isnull(game_train['first_blood'])
# On garde en memoire les game_id pour les retirer aussi de l'autre jeu
game_to_remove = np.unique(game_train['game_id'][bool])
# On ne garde que les game_id juges valable
game_train = game_train[:][~bool]

# On retire 271 matchs sur 1864, soit 15% des données. Cela fait beaucoup de pertes...

# Maintenant on veut retirer ces matchs du jeu de donnees avec les joueurs
# On va commencer par trier les donnees par game_id - team_id - player_id
player_train.sort_values(by = ['game_id', 'team_id', 'player_id'], inplace = True)
# Indices des matchs a retirer
bool = pd.Series(player_train['game_id']).isin(game_to_remove)
player_train = player_train[:][~bool]

# On retourne sur game_train.
# On veut remplacer créer une nouvelle variable valant 1 si l'équipe a gg, 0 sinon
victory = (game_train['winner_id'] == game_train['team_id'])*1
game_train['victory'] = victory

# On veut remplacer tous les T/F par 1/0 dans les deux dataset
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

game_train = replaceTFgame(game_train)
player_train = replaceTFplayer(player_train)


#------------
# Fusion des deux jeux de donnees
#------------

# On souhaite fusionner les deux jeux de donnees afin d'avoir plus d'informations pour faire nos predictions.
# Le but est de sommer les variables qui peuvent l'etre dans le jeu de donnees jouer/joueur pour en faire des variables par equipe
# On va perdre certaines informations, comme le champion utilisé par chaque joueur, mais dans ce cas ce n'est pas tres grave car
# on retrouve en general le meme type de champion dans chaque lane.


# Grace a notre façon de trier, on a juste a faire la somme de cinq elements a la suite pour avoir la somme par equipe
player_temp = player_train.drop(['game_id', 'team_id', 'player_id', 'champion_id','first_blood_kill',
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

# On range la matrice obtenu dans game_train
n = game_train.shape[0]
game_train.index = (range(0, n))
game_train = game_train.join(res)

# On veut savoir si des variables contiennent des NaN
test = game_train.apply(np.isnan)
for col in test.columns:
	print(np.unique(test[col]))


#------------
# Première analyse
#------------

# On fait quelques histogrammes

plt.hist(game_train[0], bins = 12)
plt.title("Distribution des kill")
plt.show()

plt.hist(game_train[1], bins = 12)
plt.title("Distribution des death")
plt.show()

plt.hist(game_train[2], bins = 12)
plt.title("Distribution des assists")
plt.show()

















































