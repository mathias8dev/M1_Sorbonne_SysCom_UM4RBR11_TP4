"""
Created on Thu Mar 27 10:37:24 2025
@author: cathe
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

def visualize_classifier(model, X, y):
    ax = plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='rainbow',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    plt.scatter(xx.ravel(), yy.ravel(), c=Z, s=0.1, cmap='rainbow')
    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

# ============================================================================
# I. CHARGEMENT ET VISUALISATION DES DONNÉES
# ============================================================================
print("="*70)
print("I. CHARGEMENT ET VISUALISATION DES DONNÉES")
print("="*70)

data = np.load("TP4.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])

print(f"\nNombre de points dans la base d'apprentissage: {len(X_train)}")
print(f"Nombre de points dans la base de test: {len(X_test)}")
print(f"Dimension des données: {X_train.shape[1]}")

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow')
plt.title("Données d'apprentissage")
plt.colorbar(label='Classe')
plt.show()

# ============================================================================
# II. ARBRE DE DÉCISION
# ============================================================================
print("\n" + "="*70)
print("II. ARBRE DE DÉCISION")
print("="*70)

# a. Principe des arbres de décision
print("\na. Principe des arbres de décision")
print("-" * 50)

# Remplacez *** par les 3 derniers chiffres de votre numéro d'étudiant
random_state = 123  # MODIFIEZ CETTE VALEUR

tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=random_state)
tree1.fit(X_train, y_train)

print("\nVisualisation du classifieur:")
plt.figure(figsize=(10, 8))
visualize_classifier(tree1, X_train, y_train)
plt.title("Partitionnement de l'espace (max_depth=3)")

print("\nStructure de l'arbre:")
plt.figure(figsize=(20, 10))
tree.plot_tree(tree1, filled=True, feature_names=['X1', 'X2'], class_names=[str(i) for i in np.unique(y_train)])
plt.title("Arbre de décision (max_depth=3)")
plt.show()

text_representation = tree.export_text(tree1, feature_names=['X1', 'X2'])
print("\nReprésentation textuelle de l'arbre:")
print(text_representation)

print("\nRéponses aux questions:")
print("- max_depth: représente la profondeur maximale de l'arbre")
print("- visualize_classifier: montre le partitionnement de l'espace en régions")
print("- tree.plot_tree: affiche la structure hiérarchique de l'arbre")
print("- tree.export_text: donne une représentation textuelle des règles de décision")

# b. Performance d'un classifieur multi-classes
print("\n\nb. Performance d'un classifieur multi-classes")
print("-" * 50)

y_pred = tree1.predict(X_test)
C = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion:")
print(C)
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))
print(f'Accuracy = {accuracy_score(y_test, y_pred):.4f}')

print("\nCalcul manuel pour la classe 0 (première ligne):")
if len(C) > 0:
    TP = C[0, 0]
    FP = C[:, 0].sum() - TP
    FN = C[0, :].sum() - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-score: {f1_score:.2f}")

# c. Optimisation de la profondeur de l'arbre
print("\n\nc. Optimisation de la profondeur de l'arbre")
print("-" * 50)

depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=random_state)
    tree_model.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, tree_model.predict(X_train)))
    test_scores.append(accuracy_score(y_test, tree_model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Train')
plt.plot(depths, test_scores, 's-', label='Test')
plt.xlabel('Profondeur maximale')
plt.ylabel('Taux de reconnaissance')
plt.title('Évolution du taux de reconnaissance en fonction de la profondeur')
plt.legend()
plt.grid(True)
plt.show()

best_depth = depths[np.argmax(test_scores)]
best_score = max(test_scores)
print(f"\nMeilleure profondeur: {best_depth}")
print(f"Meilleur taux de classification: {best_score:.4f}")

# Affichage du partitionnement optimal
best_tree = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, random_state=random_state)
best_tree.fit(X_train, y_train)
plt.figure(figsize=(10, 8))
visualize_classifier(best_tree, X_train, y_train)
plt.title(f'Partitionnement optimal (max_depth={best_depth})')

print("\nConclusion: Le taux augmente avec la profondeur puis se stabilise ou diminue (surapprentissage)")

# d. Arbre de décision sur des données de grande dimension
print("\n\nd. Arbre de décision sur des données de grande dimension")
print("-" * 50)

data_b = np.load("TP4b.npz")
X_train_b, y_train_b, X_test_b, y_test_b = (data_b[key] for key in ["X_train", "y_train", "X_test", "y_test"])

print(f"\nNombre de points dans la base d'apprentissage: {len(X_train_b)}")
print(f"Nombre de points dans la base de test: {len(X_test_b)}")
print(f"Dimension des données: {X_train_b.shape[1]}")

depths_b = range(1, 31)
test_scores_b = []

for depth in depths_b:
    tree_b = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=random_state)
    tree_b.fit(X_train_b, y_train_b)
    test_scores_b.append(accuracy_score(y_test_b, tree_b.predict(X_test_b)))

plt.figure(figsize=(10, 6))
plt.plot(depths_b, test_scores_b, 'o-')
plt.xlabel('Profondeur maximale')
plt.ylabel('Taux de reconnaissance')
plt.title('Taux de reconnaissance (données haute dimension)')
plt.grid(True)
plt.show()

best_depth_b = depths_b[np.argmax(test_scores_b)]
best_score_b = max(test_scores_b)
print(f"\nMeilleure profondeur: {best_depth_b}")
print(f"Meilleur taux de classification: {best_score_b:.4f}")

# ============================================================================
# III. FORÊT D'ARBRES ALÉATOIRES
# ============================================================================
print("\n\n" + "="*70)
print("III. FORÊT D'ARBRES ALÉATOIRES")
print("="*70)

# a. Test d'une forêt d'arbres aléatoires
print("\na. Test d'une forêt d'arbres aléatoires")
print("-" * 50)

RF = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=random_state)
RF.fit(X_train_b, y_train_b)

y_pred_rf = RF.predict(X_test_b)
acc_rf = accuracy_score(y_test_b, y_pred_rf)
print(f"\nTaux de classification avec 10 arbres: {acc_rf:.4f}")
print("Conclusion: La forêt améliore significativement les performances par rapport à un seul arbre")

# b. Influence des paramètres
print("\n\nb. Influence des paramètres")
print("-" * 50)

# Évolution en fonction du nombre d'arbres
print("\n1. Influence du nombre d'arbres:")
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
scores_estimators = []
times_estimators = []

for n_est in n_estimators_range:
    start = time.time()
    rf = RandomForestClassifier(criterion='entropy', n_estimators=n_est, random_state=random_state)
    rf.fit(X_train_b, y_train_b)
    elapsed = time.time() - start
    times_estimators.append(elapsed)
    scores_estimators.append(accuracy_score(y_test_b, rf.predict(X_test_b)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, scores_estimators, 'o-')
plt.xlabel('Nombre d\'arbres')
plt.ylabel('Taux de reconnaissance')
plt.title('Influence du nombre d\'arbres')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, times_estimators, 'o-')
plt.xlabel('Nombre d\'arbres')
plt.ylabel('Temps (s)')
plt.title('Temps d\'entraînement')
plt.grid(True)
plt.tight_layout()
plt.show()

best_n_estimators = n_estimators_range[np.argmax(scores_estimators)]
print(f"Meilleur compromis temps/taux: {best_n_estimators} arbres")

# Évolution en fonction de la profondeur
print("\n2. Influence de la profondeur:")
depths_rf = [3, 5, 10, 15, 20, None]
scores_depth = []

for depth in depths_rf:
    rf = RandomForestClassifier(criterion='entropy', n_estimators=best_n_estimators, 
                                max_depth=depth, random_state=random_state)
    rf.fit(X_train_b, y_train_b)
    scores_depth.append(accuracy_score(y_test_b, rf.predict(X_test_b)))

plt.figure(figsize=(10, 6))
x_labels = [str(d) if d is not None else 'None' for d in depths_rf]
plt.plot(range(len(depths_rf)), scores_depth, 'o-')
plt.xticks(range(len(depths_rf)), x_labels)
plt.xlabel('Profondeur maximale')
plt.ylabel('Taux de reconnaissance')
plt.title('Influence de la profondeur')
plt.grid(True)
plt.show()

best_depth_rf = depths_rf[np.argmax(scores_depth)]
print(f"Meilleure profondeur: {best_depth_rf}")

# Évolution en fonction du nombre de features
print("\n3. Influence du nombre de caractéristiques (max_features):")
max_features_range = [1, 2, 5, 10, 'sqrt', 'log2', None]
scores_features = []

for max_feat in max_features_range:
    rf = RandomForestClassifier(criterion='entropy', n_estimators=best_n_estimators,
                                max_depth=best_depth_rf, max_features=max_feat,
                                random_state=random_state)
    rf.fit(X_train_b, y_train_b)
    scores_features.append(accuracy_score(y_test_b, rf.predict(X_test_b)))

plt.figure(figsize=(10, 6))
x_labels_feat = [str(f) for f in max_features_range]
plt.plot(range(len(max_features_range)), scores_features, 'o-')
plt.xticks(range(len(max_features_range)), x_labels_feat, rotation=45)
plt.xlabel('max_features')
plt.ylabel('Taux de reconnaissance')
plt.title('Influence du nombre de caractéristiques')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nConclusion: Ces évolutions sont prévisibles - plus d'arbres améliore la performance,")
print("la profondeur doit être optimisée, et max_features contrôle la diversité des arbres")

# c. Choix des paramètres optimaux
print("\n\nc. Choix des paramètres optimaux avec GridSearchCV")
print("-" * 50)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20, None],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(criterion='entropy', random_state=random_state),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nRecherche des paramètres optimaux en cours...")
grid_search.fit(X_train_b, y_train_b)

print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score (validation croisée): {grid_search.best_score_:.4f}")

y_pred_grid = grid_search.best_estimator_.predict(X_test_b)
final_accuracy = accuracy_score(y_test_b, y_pred_grid)
print(f"Taux de reconnaissance sur le test: {final_accuracy:.4f}")

print("\nRéponse à la question:")
print("Le paramètre cv de GridSearchCV représente le nombre de folds pour la validation croisée.")
print("cv=5 signifie que les données sont divisées en 5 parties pour évaluer les performances.")

print("\n" + "="*70)
print("FIN DU TP")
print("="*70)