"""Tests unitaires du pipeline ML (model_pipeline.py)."""

import pytest

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"


@pytest.fixture(scope="module")
def data():
    """Prépare les données une seule fois pour tous les tests."""
    return prepare_data(DATA_PATH)


@pytest.fixture(scope="module")
def model(data):
    """Entraîne le modèle une seule fois pour tous les tests."""
    x_train, _, y_train, _ = data
    return train_model(x_train, y_train)


def test_prepare_data_shapes(data):
    """Les features et les labels doivent avoir la même taille."""
    x_train, x_test, y_train, y_test = data
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) > len(x_test)


def test_train_model_returns_model(model):
    """Le modèle entraîné doit pouvoir prédire."""
    assert hasattr(model, "predict")


def test_save_and_load_model(model, tmp_path):
    """Sauvegarder puis recharger doit donner un modèle utilisable."""
    path = tmp_path / "model_test.joblib"
    save_model(model, str(path))
    assert path.exists()
    loaded = load_model(str(path))
    assert hasattr(loaded, "predict")


def test_evaluate_model_accuracy(model, data):
    """L'accuracy doit être comprise entre 0 et 1."""
    _, x_test, _, y_test = data
    accuracy, _ = evaluate_model(model, x_test, y_test)
    assert 0 <= accuracy <= 1
