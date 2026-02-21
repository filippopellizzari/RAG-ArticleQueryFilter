"""FastAPI application for the RAG Article Query Filter.

Endpoints
---------
GET  /health   Liveness check.
POST /query    Retrieve relevant article UUIDs for a (non-toxic) query.

Usage
-----
    uvicorn src.api:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request

from src.indexing import load_index, retrieve_doc_uuids
from src.query_filter import is_query_valid


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the vector index once at startup and release on shutdown."""
    app.state.index = load_index()
    yield


app = FastAPI(
    title="RAG Article Query Filter",
    description=(
        "Retrieve relevant news article UUIDs for a query using a hybrid "
        "BM25 + dense-vector retrieval pipeline. Toxic queries are rejected."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness check — returns 200 when the service is ready."""
    return {"status": "ok"}


@app.post("/query", tags=["retrieval"])
async def handle_query(
    query: str,
    request: Request,
) -> dict[str, list[str]]:
    """Return ranked article UUIDs relevant to *query*.

    Parameters
    ----------
    query:
        Free-text search query (passed as a query-string parameter).

    Returns
    -------
    JSON object with a ``uuids`` list of ranked article identifiers.

    Raises
    ------
    422 Unprocessable Entity
        If the query is classified as harmful by the toxicity filter.
    """
    if not is_query_valid(query):
        raise HTTPException(status_code=422, detail="Harmful content detected.")

    uuids = retrieve_doc_uuids(query, request.app.state.index)
    return {"uuids": uuids}
