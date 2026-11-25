"""Atelier 3 : pipeline Prefect.

Usage :
    python pipeline_prefect.py --flow install
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow all
"""

import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

MODEL_PATH = "classifier.joblib"

# On analyse seulement nos 3 fichiers (sinon venv/ est scanné : très lent)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

TEST_FILE = Path("tests/test_model_pipeline.py")
TEST_CONTENT = '''"""Exemple de test unitaire."""
import model_pipeline


def test_functions_exist():
    """Vérifie que les 5 fonctions du pipeline existent."""
    for name in (
        "prepare_data",
        "train_model",
        "evaluate_model",
        "save_model",
        "load_model",
    ):
        assert callable(getattr(model_pipeline, name))
'''


def run_cmd(cmd, check=True):
    """Exécute une commande, affiche la sortie, lève une erreur si elle échoue."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Échec : {' '.join(cmd)}")
    return result.returncode


# ------------------------------- TASKS -------------------------------
@task(name="Installer les dépendances")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code():
    run_cmd([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="Qualité du code (flake8)")
def check_quality():
    # check=False : on affiche les problèmes sans arrêter le pipeline
    run_cmd(
        [sys.executable, "-m", "flake8", "--max-line-length=110", *PROJECT_FILES],
        check=False,
    )


@task(name="Sécurité du code (bandit)")
def check_security():
    run_cmd([sys.executable, "-m", "bandit", "-q", "-ll", *PROJECT_FILES], check=False)


@task(name="Tests unitaires (pytest)")
def run_unit_tests():
    # Crée l'exemple de test s'il n'existe pas, puis lance pytest
    if not TEST_FILE.exists():
        TEST_FILE.parent.mkdir(exist_ok=True)
        TEST_FILE.write_text(TEST_CONTENT)
    run_cmd([sys.executable, "-m", "pytest", "-q", str(TEST_FILE)])


@task(name="Préparation des données")
def prepare_data():
    run_cmd([sys.executable, "main.py", "--prepare"])


@task(name="Entraînement du modèle")
def train_model():
    # main.py --train entraîne ET sauvegarde le modèle
    run_cmd([sys.executable, "main.py", "--train"])


@task(name="Sauvegarder le modèle")
def save_model():
    # La sauvegarde est faite par --train : on vérifie que le fichier existe
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"{MODEL_PATH} introuvable")
    get_run_logger().info("Modèle sauvegardé : %s", MODEL_PATH)


@task(name="Charger le modèle")
def load_model():
    run_cmd([sys.executable, "main.py", "--load"])


@task(name="Évaluer le modèle")
def evaluate_model():
    run_cmd([sys.executable, "main.py", "--evaluate"])


# ------------------------------- FLOWS -------------------------------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    prepare_data()
    train_model()


@flow(name="evaluate")
def evaluate_flow():
    load_model()
    evaluate_model()


@flow(name="all")
def all_flow():
    install_dependencies()
    code_flow()
    prepare_data()
    train_model()
    save_model()
    evaluate_model()


FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
