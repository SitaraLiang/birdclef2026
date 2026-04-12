import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def main():
    print("=== ANALYSE EN COMPOSANTES PRINCIPALES (PCA) ===")
    
    fichier = 'X_train.csv'
    df = pd.read_csv(fichier)
    
    # les variables numériques 
    colonnes_a_ignorer = ['primary_label', 'filename', 'latitude', 'longitude']
    X = df.drop(columns=[c for c in colonnes_a_ignorer if c in df.columns])
    
    print(f"Taille originale : {X.shape[1]} variables.")

    # on est obligé de standardisé nos valeurs pour la PCA 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    #calcul de la variance expliquée cumulée
    variance_cumulee = np.cumsum(pca.explained_variance_ratio_) * 100
    
    # plots ( elbow method)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(variance_cumulee) + 1), variance_cumulee, marker='o', linestyle='--')
    
    #ajoute une ligne rouge horizontale à 95% de variance (c'est le standard, ça nous sert à déterminer visuellement le nombre X qui suffit à staisfaire cette condition)
    plt.axhline(y=95, color='r', linestyle='-')
    plt.text(5, 96, 'Seuil de 95% d\'information', color='red', fontsize=12)
    
    plt.title("PCA : Pourcentage d'information conservée par composante")
    plt.xlabel("Nombre de Composantes Principales")
    plt.ylabel("Variance expliquée cumulée (%)")
    plt.grid(True)
    plt.savefig('pca_variance.png', bbox_inches='tight')
    plt.close()
    
    # Trouver combien de composantes pour 95%
    n_components_95 = np.argmax(variance_cumulee >= 95) + 1
    print(f"Pour garder 95% de l'information, il faut {n_components_95} composantes (au lieu de {X.shape[1]}).")
    
    # crée le nouveau dataset réduit 
    print("Génération du dataset réduit...")
    pca_finale = PCA(n_components=n_components_95)
    X_pca = pca_finale.fit_transform(X_scaled)
    
    # on recree un dataframe propre
    df_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components_95)])
    df_pca['primary_label'] = df['primary_label']
    df_pca.to_csv('X_train_pca.csv', index=False)
    
    print("Terminé ! Le graphique 'pca_variance.png' et le fichier 'X_train_pca.csv' ont été générés.")

if __name__ == "__main__":
    main()