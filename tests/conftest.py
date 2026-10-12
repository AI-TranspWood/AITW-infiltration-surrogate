"""Fixtures for the aitw-infiltration-surrogate tests."""
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def dataset_file_path():
    """Fixture to provide the path to the dataset file."""
    return "examples/dataset.csv"


@pytest.fixture(scope="session")
def dataset(dataset_file_path):
    """Fixture to provide the path to the dataset."""
    return pd.read_csv(dataset_file_path)

@pytest.fixture(scope="session")
def dataset_no_sat_time():
    """Fixture to provide a dataset without the `sat_time` column for testing."""
    # (Samples x 6 shape)
    data = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ]
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def dataset_wrong_negative_values():
    """Fixture to provide a dataset with negative values for testing."""
    # (Samples x 7 shape)
    data = [
        [-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.0],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 3.0],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 4.0],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 5.0],
    ]
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def dataset_wrong_shape():
    """Fixture to provide a dataset with wrong shape for testing."""
    # (Samples x 5 shape)
    data = [
        [0.1, 0.2, 0.3, 0.4, 1.0],
        [0.2, 0.3, 0.4, 0.5, 2.0],
        [0.3, 0.4, 0.5, 0.6, 3.0],
        [0.4, 0.5, 0.6, 0.7, 4.0],
        [0.5, 0.6, 0.7, 0.8, 5.0],
    ]
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def dataset_nst_path(tmpdir_factory, dataset_no_sat_time):
    """Fixture to provide the path to the dataset without `sat_time` column."""
    file_path = tmpdir_factory.mktemp("data") / "dataset_no_sat_time.csv"
    dataset_no_sat_time.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture(scope="session")
def dataset_wnv_path(tmpdir_factory, dataset_wrong_negative_values):
    """Fixture to provide the path to the dataset with negative values."""
    file_path = tmpdir_factory.mktemp("data") / "dataset_wrong_negative_values.csv"
    dataset_wrong_negative_values.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture(scope="session")
def dataset_ws_path(tmpdir_factory, dataset_wrong_shape):
    """Fixture to provide the path to the dataset with wrong shape."""
    file_path = tmpdir_factory.mktemp("data") / "dataset_wrong_shape.csv"
    dataset_wrong_shape.to_csv(file_path, index=False)
    return str(file_path)
