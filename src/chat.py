from dotenv import load_dotenv
import os
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient

# Load environment variables
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

# Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="tree_summarize"
)

# Query
instructions = "You are a helpful assistant that can recommend articles on a given topic. You can recommend up to 5 articles. Do not recommend the same url twice. Always provide title, substack, short description of the postand url."
q = "Do you know any articles about Ethan Mollick and prediction for AI in 2025"
response = query_engine.query(instructions + "\n\n" + q)
print("Response:", response)

# Print sources
print("\nSources:")
for source_node in response.source_nodes:
    print(f"\nSource Metadata: {source_node.metadata}")
    print(f"Source Text: {source_node.text}...")