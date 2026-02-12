import pytest
from pathlib import Path


@pytest.fixture
def data_dir() -> Path:
    """Return the absolute path to the directory containing test data."""
    return Path(__file__).parent / "data"
