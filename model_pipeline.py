"""
model_pipeline.py
-----------------
Fonctions modulaires du pipeline Machine Learning (prédiction du churn bancaire).
Chaque fonction correspond à une étape du notebook Jupyter.
"""

# ===================== IMPORTS =====================
import os  # pour vérifier l'existence des fichiers
import joblib  # pour sauvegarder / charger le modèle
import pandas as pd  # pour manipuler les données (DataFrame)
from sklearn.ensemble import RandomForestClassifier  # le modèle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # métriques
from sklearn.model_selection import train_test_split  # découpage train/test


# ===================== 1. PRÉPARATION DES DONNÉES =====================
def prepare_data(csv_path, test_size=0.2, random_state=1):
    """
    Charge le CSV, nettoie les données et sépare train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion des données réservée au test (0.2 = 20 %).
        random_state (int): graine aléatoire pour obtenir toujours le même découpage.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    # Vérifier que le fichier existe avant de le lire
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(csv_path)

    # Nettoyage : supprimer les doublons et les lignes avec valeurs manquantes
    df = df.drop_duplicates()
    df = df.dropna()

    # Encodage : transformer Gender (texte) en nombres (Female=0, Male=1)
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # Séparer la cible (y = Exited) des variables explicatives (x)
    y = df["Exited"]
    # On enlève les colonnes inutiles (identifiants, texte) et la cible
    x = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography", "Exited"])

    # Découper en jeu d'entraînement (80 %) et jeu de test (20 %)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    print(f"[prepare_data] Train : {x_train.shape} | Test : {x_test.shape}")
    return x_train, x_test, y_train, y_test


# ===================== 2. ENTRAÎNEMENT DU MODÈLE =====================
def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un modèle Random Forest.

    Args:
        x_train (DataFrame): variables explicatives d'entraînement.
        y_train (Series): cible d'entraînement.
        n_estimators (int): nombre d'arbres dans la forêt.
        random_state (int): graine aléatoire pour la reproductibilité.

    Returns:
        RandomForestClassifier: le modèle entraîné.
    """
    # Créer le modèle avec ses paramètres
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(x_train, y_train)

    print("[train_model] Modèle entraîné.")
    return model


# ===================== 3. ÉVALUATION DU MODÈLE =====================
def evaluate_model(model, x_test, y_test):
    """
    Évalue le modèle sur le jeu de test et affiche les métriques.

    Args:
        model: modèle entraîné.
        x_test (DataFrame): variables explicatives de test.
        y_test (Series): vraies valeurs de la cible.

    Returns:
        tuple: (accuracy, matrice de confusion)
    """
    # Prédire sur les données de test
    y_pred = model.predict(x_test)

    # Calculer l'accuracy (proportion de bonnes prédictions)
    accuracy = accuracy_score(y_test, y_pred)

    # Matrice de confusion : bonnes et mauvaises prédictions par classe
    matrix = confusion_matrix(y_test, y_pred)

    # Afficher les résultats
    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("Matrice de confusion :")
    print(matrix)
    print(classification_report(y_test, y_pred))

    return accuracy, matrix


# ===================== 4. SAUVEGARDE DU MODÈLE =====================
def save_model(model, path="classifier.joblib"):
    """
    Sauvegarde le modèle entraîné sur le disque avec joblib.

    Args:
        model: modèle à sauvegarder.
        path (str): nom du fichier de sortie.
    """
    # Écrire le modèle dans un fichier .joblib
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


# ===================== 5. CHARGEMENT DU MODÈLE =====================
def load_model(path="classifier.joblib"):
    """
    Charge un modèle précédemment sauvegardé.

    Args:
        path (str): chemin du fichier du modèle.

    Returns:
        Le modèle chargé, prêt à faire des prédictions.
    """
    # Vérifier que le fichier existe, sinon afficher une erreur claire
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    # Lire le modèle depuis le fichier
    model = joblib.load(path)

    print(f"[load_model] Modèle chargé depuis {path}")
    return model
