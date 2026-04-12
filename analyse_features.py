import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os


# À NOTER : J'utilise un xgboost à seul but d'analyse des features, je ne teste pas ce mini model sur train_soundscape, c'est pas le but ici 

def main():
    print("=== ANALYSE DES FEATURES AUDIO ===")
    
   
    fichier_entree = 'X_train.csv' 
    if not os.path.exists(fichier_entree):
        print(f" Fichier {fichier_entree} introuvable.")
        return
        
    df = pd.read_csv(fichier_entree)
    print(f"Données chargées : {df.shape[0]} exemples, {df.shape[1]} colonnes.")
    
    # On sépare les features (X) de la cible (y)
    colonnes_a_ignorer = ['primary_label', 'filename', 'latitude', 'longitude']
    X = df.drop(columns=[c for c in colonnes_a_ignorer if c in df.columns])
    y = df['primary_label']
    
    # analyse bivariee
    print("Génération de la matrice de corrélation...")
    plt.figure(figsize=(20, 16))
    # On calcule la corrélation de Pearson
    corr = X.corr()
    # On masque la moitié supérieure (en miroir) pour la lisibilité
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Matrice de Corrélation des Features Audio", fontsize=16)
    plt.savefig('correlation_features.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # analyse multivariee
    print("Entraînement d'un Random Forest pour évaluer l'importance des variables...")

    # On transforme les noms d'espèces (ex: 1161364) en numéros simples (0,1, 2...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # On entraîne un modèle d'évaluation rapide (utilise tous les coeurs : n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    # On récupère le score d'importance de chaque colonne
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))

    # On n'affiche que le Top 30 pour que ça reste lisible
    sns.barplot(x=importances.head(30).values, y=importances.head(30).index, palette='viridis', hue=importances.head(30).index, legend=False)
    plt.title("Top 30 des Features les plus importantes (Random Forest)", fontsize=16)
    plt.xlabel("Importance (Gini)")
    plt.ylabel("Features")
    plt.savefig('importance_features.png', bbox_inches='tight', dpi=150)
    plt.close()

    print("Terminé ! Deux images ont été générées :")
    print(" - correlation_features.png")
    print(" - importance_features.png")

if __name__ == "__main__":
    main()
