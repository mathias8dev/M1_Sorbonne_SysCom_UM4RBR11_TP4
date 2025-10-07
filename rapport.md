# TP4 - Forêt d'arbres aléatoires


## I. Chargement et visualisation des données

J'ai chargé les données du fichier TP4.npz. Voici ce que j'ai trouvé :

- Base d'apprentissage : [X] points
- Base de test : [Y] points  
- Dimension : 2

Les données sont en 2D donc on peut les visualiser facilement. Sur le graphique on voit plusieurs classes de couleurs différentes. Les points sont assez mélangés donc la classification va pas être simple.

## II. Arbre de décision

### II.a - Principe

J'ai créé un premier arbre avec `max_depth=3`. 

**max_depth** : c'est la profondeur maximale de l'arbre, ça contrôle combien de fois on peut découper les données.

Les 3 fonctions donnent des infos complémentaires :
- `visualize_classifier` : montre comment l'espace est découpé en régions de couleurs
- `tree.plot_tree` : affiche la structure de l'arbre avec les nœuds et les conditions
- `tree.export_text` : donne les règles de décision en format texte

Sur la figure, on retrouve bien toutes les découpes verticales et horizontales qui correspondent aux conditions dans l'arbre.

### II.b - Performance

J'ai testé l'arbre sur la base de test. Le classification_report donne pour chaque classe :
- **precision** : parmi ce que le modèle a prédit comme classe X, combien sont vraiment de classe X
- **recall** : parmi tous les vrais X, combien le modèle en a trouvé
- **f1-score** : moyenne harmonique entre precision et recall
- **support** : nombre d'exemples de cette classe

**Vérification calcul manuel pour classe 0 :**

À partir de la matrice de confusion, j'ai calculé :
- TP (vrais positifs) = C[0,0]
- FP (faux positifs) = somme colonne 0 - TP
- FN (faux négatifs) = somme ligne 0 - TP

Puis :
- precision = TP/(TP+FP) = [valeur]
- recall = TP/(TP+FN) = [valeur]
- f1-score = 2×precision×recall/(precision+recall) = [valeur]

Ça correspond bien aux valeurs du classification_report.

### II.c - Optimisation de la profondeur

J'ai testé différentes valeurs de max_depth entre 1 et 20.

**Observations :**
- Au début, quand on augmente la profondeur, le taux de reconnaissance augmente
- Ça monte jusqu'à environ [X] de profondeur
- Après ça stagne ou même ça diminue un peu

**Explication :** quand l'arbre est trop profond, il fait du surapprentissage (overfitting). Il apprend trop bien les données d'entraînement et généralise moins bien sur le test.

Meilleure profondeur trouvée : [X] avec un taux de [Y]%

Le partitionnement est plutôt satisfaisant, on voit bien les différentes zones de couleur qui correspondent aux classes. Par contre dans certaines zones il y a encore des erreurs, surtout là où les classes se mélangent beaucoup.

### II.d - Données de grande dimension

Avec TP4b.npz :
- Points train : [X]
- Points test : [Y]
- Dimension : [Z] (beaucoup plus grand !)

J'ai refait l'optimisation de la profondeur. Cette fois on peut pas visualiser les données vu qu'on est en dimension > 2.

Meilleur taux obtenu : [X]%

C'est moins bon que sur les données 2D, ce qui est normal car le problème est plus complexe.

## III. Forêt d'arbres aléatoires

### III.a - Test de base

J'ai testé un RandomForestClassifier avec 10 arbres sur TP4b.

Résultat : [X]% d'accuracy

**Conclusion :** La forêt marche beaucoup mieux qu'un seul arbre ! C'est logique parce que chaque arbre vote et ça fait une moyenne des décisions, donc c'est plus robuste.

### III.b - Influence des paramètres

**1. Nombre d'arbres**

J'ai testé de 1 à 200 arbres. Le taux de reconnaissance augmente avec le nombre d'arbres mais à partir d'un certain point (environ [X] arbres) ça se stabilise. Le temps d'exécution augmente de façon linéaire.

**Meilleur compromis temps/taux :** [X] arbres

C'est un bon équilibre entre performance et temps de calcul.

**2. Profondeur maximale**

En gardant le nombre d'arbres optimal, j'ai testé différentes profondeurs. 

Résultats : la profondeur optimale est [X]. Si on met None (pas de limite), ça marche bien aussi mais ça risque l'overfitting.

**3. Nombre de features (max_features)**

Ce paramètre contrôle combien de features on utilise pour chaque split dans les arbres. Ça permet d'avoir plus de diversité entre les arbres.

Résultats : [X] features donne les meilleurs résultats.

**Est-ce prévisible ?** Oui en partie :
- Plus d'arbres = meilleur (jusqu'à convergence) ✓
- Profondeur à optimiser pour éviter overfitting ✓
- max_features contrôle la diversité, donc pas trop bas ni trop haut ✓

### III.c - GridSearchCV

GridSearchCV teste automatiquement plein de combinaisons de paramètres et trouve la meilleure.

**Paramètre cv :** c'est le nombre de "folds" pour la validation croisée. Avec cv=5, les données sont divisées en 5 parties. On entraîne sur 4 parties et on teste sur la 5ème, et on fait ça 5 fois en changeant quelle partie on utilise pour tester. Ça permet d'avoir une meilleure estimation des performances.

**Meilleurs paramètres trouvés :**
- n_estimators : [X]
- max_depth : [Y]
- max_features : [Z]

**Taux final sur le test :** [X]%

## Conclusion générale

Dans ce TP j'ai pu voir que :

1. Les arbres de décision marchent bien mais ont tendance à faire de l'overfitting si on les laisse trop profonds

2. Les forêts d'arbres aléatoires (Random Forest) améliorent beaucoup les performances en combinant plusieurs arbres. C'est plus stable et généralement meilleur qu'un seul arbre.

3. Il faut optimiser les hyperparamètres. GridSearchCV est très pratique pour ça, même si ça prend du temps de calcul.

4. Sur des données de haute dimension, la classification est plus difficile mais les Random Forests restent efficaces.

Les Random Forests sont un bon algorithme pour la classification : robustes, performants, et pas trop compliqués à utiliser.