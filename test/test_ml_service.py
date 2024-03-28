import pytest
from src.data.data_loader import DataLoader, MockDataLoader
from src.ml_service import (
    prepare_data,
)


@pytest.fixture
def loader():
    return MockDataLoader(1000, 500)


def test_prepare_data_splits(loader: DataLoader):
    total_samples = loader.load_raw_data()[0].shape[0]
    validation_size = 0.2
    test_size = 0.25

    X_train, X_validate, X_test, y_train, y_validate, y_test = prepare_data(
        validation_size, test_size, loader
    )

    # Calculate expected sizes
    expected_validation_size = int(total_samples * validation_size)
    expected_test_size = int(total_samples * test_size)
    expected_train_size = total_samples - expected_validation_size - expected_test_size

    # Validate the sizes
    assert len(X_train) == expected_train_size, "Incorrect training set size"
    assert len(X_validate) == expected_validation_size, "Incorrect validation set size"
    assert len(X_test) == expected_test_size, "Incorrect test set size"

    assert len(y_train) == expected_train_size, "Incorrect training set size"
    assert len(y_validate) == expected_validation_size, "Incorrect validation set size"
    assert len(y_test) == expected_test_size, "Incorrect test set size"


def test_prepare_data_validation_size(loader: DataLoader):
    with pytest.raises(ValueError):
        prepare_data(validation_size=-0.1, test_size=0.25, loader=loader)

    with pytest.raises(ValueError):
        prepare_data(validation_size=1.1, test_size=0.25, loader=loader)


def test_prepare_data_test_size(loader: DataLoader):
    with pytest.raises(ValueError):
        prepare_data(validation_size=0.2, test_size=-0.1, loader=loader)

    with pytest.raises(ValueError):
        prepare_data(validation_size=0.2, test_size=1.1, loader=loader)


def test_prepare_data_holdout_size(loader: DataLoader):
    with pytest.raises(ValueError):
        prepare_data(validation_size=0.8, test_size=0.25, loader=loader)

    with pytest.raises(ValueError):
        prepare_data(validation_size=0.2, test_size=0.8, loader=loader)


def test_prepare_data_with_no_holdout_data(loader: DataLoader):
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(
        validation_size=0.0, test_size=0.0, loader=loader
    )
    total_samples = loader.load_raw_data()[0].shape[0]
    assert len(x_train) == total_samples
    assert len(x_val) == 0
    assert len(x_test) == 0
    assert len(y_train) == total_samples
    assert len(y_val) == 0
    assert len(y_test) == 0


def test_prepare_data_with_no_test_data(loader: DataLoader):
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(
        validation_size=0.2, test_size=0.0, loader=loader
    )
    total_samples = loader.load_raw_data()[0].shape[0]
    expected_train_size = int(total_samples * 0.8)
    expected_validation_size = int(total_samples * 0.2)
    assert len(x_train) == expected_train_size
    assert len(x_val) == expected_validation_size
    assert len(x_test) == 0
    assert len(y_train) == expected_train_size
    assert len(y_val) == expected_validation_size
    assert len(y_test) == 0


def test_prepare_data_with_no_validation_data(loader: DataLoader):
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(
        validation_size=0.0, test_size=0.2, loader=loader
    )
    total_samples = loader.load_raw_data()[0].shape[0]
    expected_train_size = int(total_samples * 0.8)
    expected_test_size = int(total_samples * 0.2)
    assert len(x_train) == expected_train_size
    assert len(x_val) == 0
    assert len(x_test) == expected_test_size
    assert len(y_train) == expected_train_size
    assert len(y_val) == 0
    assert len(y_test) == expected_test_size
