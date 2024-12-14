"""
# Basic usage with defaults
python embed.py embed

# Specify parameters
python src/embed.py embed \
    --data-path="data/post_contents_by_newsletter.pkl" \
    --embedding-model="openai" \
    --recreate-db=True \
    --chunk-size=1024 \
    --chunk-overlap=50 \
    --collection-name="substack-search-v2-openai" \
    --newsletter-limit=2 

# Get help
python embed.py embed --help
"""
import fire
import os
import pickle
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import trafilatura
import pandas as pd
from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, Settings
import nest_asyncio

class NewsletterEmbedder:
    def __init__(self):
        load_dotenv()
        nest_asyncio.apply()
        pd.set_option("display.max_columns", None)
        
    def _get_article(self, data: dict, blog: str, idx: int) -> tuple:
        articles = data[blog]
        article_slugs = list(articles.keys())
        article = articles[article_slugs[idx]]

        metadata = {
            "title": article["title"],
            "description": article.get("description", ""),
            "cover_image": article.get("cover_image"),
            "post_date": article.get("post_date"),
            "reaction_count": article.get("reaction_count", 0),
            "comment_count": article.get("comment_count", 0),
            "audience": article["audience"],
            "canonical_url": article["canonical_url"],
            "slug": article["slug"],
            "author": {
                "name": article.get("publishedBylines", [{}])[0].get("name"),
                "photo_url": article.get("publishedBylines", [{}])[0].get("photo_url"),
            } if article.get("publishedBylines") else {}
        }

        return (
            article["id"],
            article["title"],
            article["audience"],
            article["canonical_url"],
            trafilatura.extract(article["body_html"]),
            metadata
        )

    def embed(
        self,
        data_path: str = "data/post_contents_by_newsletter.pkl",
        embedding_model: str = "openai",
        recreate_db: bool = True,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        newsletter_limit: int = None,
        collection_name: str = None
    ):
        """
        Embed newsletter articles into a vector database.
        
        Args:
            data_path: Path to the pickle file containing newsletter data
            embedding_model: Choice of embedding model ('openai' or 'mixedbread')
            recreate_db: Whether to recreate the database
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            newsletter_limit: Limit number of newsletters to process (None for all)
            collection_name: Override collection name from env variable
        """
        # Load data
        with open(data_path, "rb") as f:
            posts_dict = pickle.load(f)

        # Print newsletter info
        print("\nNewsletter subdomains and post counts:")
        print("-" * 40)
        for subdomain, posts in posts_dict.items():
            print(f"{subdomain}: {len(posts)} posts")
        print("-" * 40)
        print(f"Total newsletters: {len(posts_dict)}")

        # Set up embedding model
        if embedding_model == "openai":
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-large",
                embed_batch_size=100
            )
        elif embedding_model == "mixedbread":
            embed_model = MixedbreadAIEmbedding(
                api_key=os.environ["MXBAI_API_KEY"],
                model_name="mixedbread-ai/mxbai-embed-large-v1"
            )
        else:
            raise ValueError("Invalid embedding model choice")

        # Set up Qdrant
        collection_name = collection_name or os.environ["QDRANT_COLLECTION_NAME"]
        client = QdrantClient(
            os.environ["QDRANT_URI"],
            api_key=os.environ["QDRANT_API_KEY"]
        )

        # Handle collection creation/recreation
        if recreate_db or not client.collection_exists(collection_name):
            if client.collection_exists(collection_name) and recreate_db:
                print("Deleting collection...")
                client.delete_collection(collection_name=collection_name)
            print("Creating collection...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=3072,
                    distance=models.Distance.COSINE
                ),
            )

        # Set up vector store
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=client,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Prepare documents
        documents = []
        if newsletter_limit:
            posts_dict = {k: posts_dict[k] for k in list(posts_dict.keys())[:newsletter_limit]}

        print(f"\nProcessing articles...")
        for blog_name, posts in posts_dict.items():
            print(f"\nProcessing {blog_name} ({len(posts)} posts)")
            for post_idx in range(len(posts)):
                try:
                    id, title, audience, canonical_url, text, metadata = self._get_article(
                        posts_dict, blog_name, post_idx
                    )
                    description = posts[list(posts.keys())[post_idx]].get("description", "")
                    text = f"Title: {title}\nDescription: {description}\nText: {text}"
                    documents.append(Document(text=text, metadata=metadata))
                except Exception as e:
                    print(f"  Error processing post {post_idx} from {blog_name}: {str(e)}")

        # Configure settings and create index
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        print("Creating vector index...")
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
        
        print("Embedding complete!")

def main():
    fire.Fire(NewsletterEmbedder)

if __name__ == "__main__":
    main()
