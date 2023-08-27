from chromadb.config import Settings, System
from chromadb.api.fastapi import FastAPI

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Chroma Client API")

# Initialize the Chroma client. You might want to move this to a configuration function.
settings = Settings(chroma_server_host="localhost", chroma_server_http_port=8000)
system = System(settings=settings)
chroma_client = FastAPI(system=system)

# Define pydantic models for request and response validation
class CreateCollectionInput(BaseModel):
    name: str
    metadata: dict = None

class CollectionOutput(BaseModel):
    id: str
    name: str
    metadata: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to Chroma Client API"}

@app.get("/collections", response_model=list[CollectionOutput])
def list_collections():
    collections = chroma_client.list_collections()
    return [CollectionOutput(id=str(coll.id), name=coll.name, metadata=coll.metadata) for coll in collections]

@app.post("/collections", response_model=CollectionOutput)
def create_collection(data: CreateCollectionInput):
    try:
        collection = chroma_client.create_collection(name=data.name, metadata=data.metadata)
        return CollectionOutput(id=str(collection.id), name=collection.name, metadata=collection.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8080)
