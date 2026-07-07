import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope="module")
def test_client():
    """
    Creates an isolated client instance for executing fast API endpoint requests.
    """
    with TestClient(app) as client:
        yield client