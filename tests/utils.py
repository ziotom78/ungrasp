from pathlib import Path


def get_reference_file_path(filename: str) -> Path:
    """Return the path to a test file"""
    return Path(__file__).parent / "data" / filename
