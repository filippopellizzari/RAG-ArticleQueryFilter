"""Integration tests for the FastAPI application.

All heavy dependencies (ChromaDB index, BERT classifier) are mocked so the
test suite runs without a pre-built index or GPU.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app

MOCK_UUIDS = ["uuid-article-1", "uuid-article-2"]


@pytest.fixture
def client():
    """TestClient with a mocked index loaded via the lifespan."""
    mock_index = MagicMock()
    with patch("src.api.load_index", return_value=mock_index):
        with TestClient(app) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_body(self, client):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ── /query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_valid_query_returns_uuids(self, client):
        with (
            patch("src.api.is_query_valid", return_value=True),
            patch("src.api.retrieve_doc_uuids", return_value=MOCK_UUIDS),
        ):
            response = client.post("/query", params={"query": "Premier League results"})

        assert response.status_code == 200
        assert response.json() == {"uuids": MOCK_UUIDS}

    def test_toxic_query_returns_422(self, client):
        with patch("src.api.is_query_valid", return_value=False):
            response = client.post("/query", params={"query": "harmful content"})

        assert response.status_code == 422
        assert "Harmful content" in response.json()["detail"]

    def test_missing_query_param_returns_422(self, client):
        response = client.post("/query")
        assert response.status_code == 422

    def test_retrieve_doc_uuids_called_with_correct_args(self, client):
        mock_index = client.app.state.index
        with (
            patch("src.api.is_query_valid", return_value=True),
            patch("src.api.retrieve_doc_uuids", return_value=MOCK_UUIDS) as mock_retrieve,
        ):
            client.post("/query", params={"query": "football scores"})

        mock_retrieve.assert_called_once_with("football scores", mock_index)

    def test_is_query_valid_not_called_when_param_missing(self, client):
        with patch("src.api.is_query_valid") as mock_valid:
            client.post("/query")
        mock_valid.assert_not_called()
