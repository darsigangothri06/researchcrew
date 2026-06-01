import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"


def test_research_missing_query(client):
    response = client.post("/research", json={"query": ""})
    assert response.status_code == 422


def test_research_short_query(client):
    response = client.post("/research", json={"query": "ab"})
    assert response.status_code == 422
