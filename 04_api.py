from fastapi import FastAPI, HTTPException

from src.backend import is_query_valid, load_chroma_index, retrieve_doc_uuids

chroma_index = load_chroma_index()

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.post("/query/")
async def handle_query(query: str) -> dict[str, list[str]]:
    # if content of query is toxic, return Bad Request
    if not is_query_valid(query):
        raise HTTPException(status_code=422, detail="Harmful content")
    uuids = retrieve_doc_uuids(query, chroma_index)
    print(uuids)
    return {"response": uuids}
