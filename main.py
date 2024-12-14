from dotenv import load_dotenv
import os
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from fastapi.responses import StreamingResponse, JSONResponse
import json
import logfire
from dotenv import load_dotenv
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.response_synthesizers import get_response_synthesizer

load_dotenv()

# Create FastAPI app
app = FastAPI()

# Set up logging
logfire.configure(
    token=os.environ["LOGFIRE_API_KEY"],
    service_name="substack-search"
)
logfire.instrument_fastapi(app)


# Create Pydantic model for request
class QueryRequest(BaseModel):
    query: str
    instructions: Optional[str] = "You are a helpful assistant that can recommend articles on a given topic. You can recommend up to 5 articles. Do not recommend the same url twice. Always provide title, substack, short description of the post and url."

# Load environment variables and initialize components
load_dotenv()

# Initialize embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    embed_batch_size=100
)

# Connect to Qdrant
client = QdrantClient(os.environ["QDRANT_URI"], api_key=os.environ["QDRANT_API_KEY"])

# Set up vector store
vector_store = QdrantVectorStore(
    collection_name=os.environ["QDRANT_COLLECTION_NAME"],
    client=client,
)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load the existing index
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# Create regular query engine
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="tree_summarize"
)

# Create streaming query engine
streaming_query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="tree_summarize",
    streaming=True  # Enable streaming mode
)

@app.post("/query")
async def query_articles(request: QueryRequest):
    try:
        full_query = f"{request.instructions}\n\n{request.query}"
        response = query_engine.query(full_query)
        
        sources = [{
            "metadata": source_node.metadata,
            "text": source_node.text
        } for source_node in response.source_nodes]
        
        return JSONResponse({
            "response": str(response),
            "sources": sources
        })
    except Exception as e:
        logfire.error("Error processing query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_articles_stream(request: QueryRequest):
    try:
        full_query = f"{request.instructions}\n\n{request.query}"
        response = streaming_query_engine.query(full_query)

        async def generate_stream() -> AsyncGenerator[bytes, None]:
            try:
                # Get the generator from the response
                response_gen = response.response_gen
                
                # Stream the response text
                for text_chunk in response_gen:
                    yield f"data: {json.dumps({'response': text_chunk})}\n\n".encode('utf-8')
                
                # Send sources after the response is complete
                sources = [{
                    "metadata": source_node.metadata,
                    "text": source_node.text
                } for source_node in response.source_nodes]
                
                yield f"data: {json.dumps({'sources': sources})}\n\n".encode('utf-8')
            except Exception as e:
                logfire.error("Error in stream generation", error=str(e))
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logfire.error("Error initializing stream", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        # Add basic health checks for dependencies
        client.get_collection(os.environ["QDRANT_COLLECTION_NAME"])
        return {"status": "healthy", "message": "All systems operational"}
    except Exception as e:
        logfire.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)