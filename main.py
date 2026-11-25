"""
main.py
-------
Point d'entrée du projet : exécute les étapes du pipeline via des arguments CLI.

Exemples d'utilisation :
    python main.py --all        (pipeline complet : les 5 étapes + décision finale)
    python main.py --prepare
    python main.py --train
    python main.py --load
    python main.py --evaluate
"""

# Module standard pour lire les arguments de la ligne de commande
import argparse

# Importer les fonctions du pipeline
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# Chemins des fichiers (modifiables ici en un seul endroit)
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"

# Seuil d'accuracy minimal pour accepter le modèle (80 %)
SEUIL_ACCURACY = 0.80


def main():
    # Définir les arguments acceptés en ligne de commande
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn Modelling")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument(
        "--train", action="store_true", help="Entraîner puis sauvegarder le modèle"
    )
    parser.add_argument(
        "--load", action="store_true", help="Charger le modèle sauvegardé"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    parser.add_argument(
        "--all", action="store_true", help="Exécuter les 5 étapes du pipeline"
    )
    args = parser.parse_args()

    # Si aucun argument n'est donné, afficher l'aide et s'arrêter
    if not (args.prepare or args.train or args.load or args.evaluate or args.all):
        parser.print_help()
        return

    # ÉTAPE 1 : préparation des données (nécessaire pour toutes les étapes)
    print("\n===== ÉTAPE 1/5 : Préparation des données =====")
    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)

    # ÉTAPES 2 et 3 : entraînement puis sauvegarde du modèle
    if args.train or args.all:
        print("\n===== ÉTAPE 2/5 : Entraînement du modèle =====")
        model = train_model(x_train, y_train)

        print("\n===== ÉTAPE 3/5 : Sauvegarde du modèle =====")
        save_model(model, MODEL_PATH)

    # ÉTAPE 4 : chargement du modèle sauvegardé
    if args.load or args.evaluate or args.all:
        print("\n===== ÉTAPE 4/5 : Chargement du modèle =====")
        model = load_model(MODEL_PATH)

    # ÉTAPE 5 : évaluation du modèle sur le jeu de test
    if args.evaluate or args.all:
        print("\n===== ÉTAPE 5/5 : Évaluation du modèle =====")
        accuracy, _ = evaluate_model(model, x_test, y_test)

        # Résumé final avec l'accuracy
        print(f"\n>>> Pipeline terminé. Accuracy finale : {accuracy * 100:.2f} %")

        # Décision finale : comparer l'accuracy au seuil fixé
        print("\n===== DÉCISION FINALE =====")
        if accuracy >= SEUIL_ACCURACY:
            print(
                f"Modèle ACCEPTÉ : accuracy {accuracy * 100:.2f} % >= seuil {SEUIL_ACCURACY * 100:.0f} %"
            )
            print("Le modèle peut être utilisé / déployé.")
        else:
            print(
                f"Modèle REFUSÉ : accuracy {accuracy * 100:.2f} % < seuil {SEUIL_ACCURACY * 100:.0f} %"
            )
            print("Le modèle doit être amélioré avant utilisation.")


# Ce bloc s'exécute seulement si on lance directement : python main.py
if __name__ == "__main__":
    main()
