import numpy as np

from workflow import split_dataset


def _make_dataset(n_samples: int, n_features: int):
    """
    Génère un petit dataset synthétique avec deux classes équilibrées
    pour tester la fonction split_dataset.
    """
    X = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    # alternance 0/1 pour garantir la stratification
    y = np.tile([0, 1], n_samples // 2)
    if len(y) < n_samples:
        y = np.append(y, 0)
    return {"X": X, "y": y}


def test_split_dataset_respects_requested_sizes():
    n_samples = 100
    n_features = 6
    test_size = 0.2
    expected_test = int(n_samples * test_size)  # 20
    expected_train = n_samples - expected_test

    data = _make_dataset(n_samples, n_features)

    split = split_dataset(data, test_size=test_size, random_state=0)

    assert split["X_train"].shape == (expected_train, n_features)
    assert split["X_test"].shape == (expected_test, n_features)
    assert split["y_train"].shape == (expected_train,)
    assert split["y_test"].shape == (expected_test,)
    # vérifie que la taille totale est conservée
    assert len(split["X_train"]) + len(split["X_test"]) == n_samples


def test_split_dataset_handles_other_test_size():
    n_samples = 120
    n_features = 4
    test_size = 0.3
    expected_test = int(n_samples * test_size)  # 36
    expected_train = n_samples - expected_test

    data = _make_dataset(n_samples, n_features)

    split = split_dataset(data, test_size=test_size, random_state=123)

    assert split["X_train"].shape == (expected_train, n_features)
    assert split["X_test"].shape == (expected_test, n_features)
    assert len(split["y_train"]) == expected_train
    assert len(split["y_test"]) == expected_test
    # Vérifie que les index ne se chevauchent pas en termes d'effectifs
    assert split["X_train"].shape[0] != split["X_test"].shape[0]

