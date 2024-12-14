"""
Example usage:

python embed_posts.py embed_posts \
    --posts_json_path=data/post_contents_by_newsletter.pkl \
    --collection_name=post-collection-v1 \
    --embedding_model=openai  # or 'mixedbread' \
    --use_existing_embeddings=True  # Set to False to create new embeddings
"""
import fire
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import numpy as np
import trafilatura
import pickle
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from tqdm import tqdm
import tiktoken

class PostEmbedder:
    def __init__(self):
        load_dotenv()
        self.client = QdrantClient(
            os.environ["QDRANT_URI"], 
            api_key=os.environ["QDRANT_API_KEY"]
        )
        # Initialize tokenizer for counting tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Used by text-embedding-ada-002
        self.max_tokens = 8192  # OpenAI's token limit
        
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            text = self.tokenizer.decode(tokens)
        return text
        
    def create_post_level_collection(self, collection_name: str):
        """Create a new collection for post-level embeddings if it doesn't exist."""
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE
            ),
        )

    def batch_embed_texts(self, texts: list[str], embed_model) -> list:
        """Embed a batch of texts, truncating if necessary."""
        # Truncate each text to fit within token limit
        truncated_texts = [self.truncate_text(text) for text in texts]
        return embed_model.get_text_embedding_batch(truncated_texts)

    def get_existing_embeddings(self, post_id: str, chunk_collection_name: str) -> list:
        """Fetch and average existing chunk embeddings for a post."""
        # Search for all chunks belonging to this post
        chunks = self.client.scroll(
            collection_name=chunk_collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=post_id)
                    )
                ]
            )
        )[0]  # [0] gets just the points, ignoring the next_page_offset

        if not chunks:
            return None

        # Extract and average embeddings
        embeddings = [chunk.vector for chunk in chunks]
        return np.mean(embeddings, axis=0).tolist()

    def get_article(self, data: dict, blog: str, idx: int) -> tuple:
        """Extract article information from the data dictionary."""
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

    def embed_posts(
        self, 
        posts_json_path: str, 
        collection_name: str, 
        embedding_model: str = "openai",
        use_existing_embeddings: bool = False,
        chunk_collection_name: str = None,
        batch_size: int = 32
    ):
        """
        Create averaged embeddings for each post and store in Qdrant.
        
        Args:
            posts_json_path: Path to the pickle file containing posts
            collection_name: Name of the Qdrant collection to create/update
            embedding_model: Choice of embedding model ('openai' or 'mixedbread')
            use_existing_embeddings: Whether to use existing chunk embeddings
            chunk_collection_name: Name of the collection containing chunk embeddings
            batch_size: Size of batches for embedding requests
        """
        # Load posts
        with open(posts_json_path, 'rb') as f:
            posts_dict = pickle.load(f)
        
        # Set up embedding model
        if embedding_model == "openai":
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-large",
                embed_batch_size=batch_size
            )
        elif embedding_model == "mixedbread":
            embed_model = MixedbreadAIEmbedding(
                api_key=os.environ["MXBAI_API_KEY"],
                model_name="mixedbread-ai/mxbai-embed-large-v1"
            )
        else:
            raise ValueError("Invalid embedding model choice")
        
        self.create_post_level_collection(collection_name)
        
        # Calculate total number of posts
        total_posts = sum(len(posts) for posts in posts_dict.values())
        
        # Process posts in batches
        current_batch = []
        batch_metadata = []
        
        with tqdm(total=total_posts, desc="Processing posts") as pbar:
            for blog_name, posts in posts_dict.items():
                for post_idx in range(len(posts)):
                    id, title, audience, canonical_url, text, metadata = self.get_article(
                        posts_dict, blog_name, post_idx
                    )
                    
                    if use_existing_embeddings:
                        # Use existing chunk embeddings
                        embedding = self.get_existing_embeddings(id, chunk_collection_name)
                        if embedding:
                            self.client.upsert(
                                collection_name=collection_name,
                                points=[models.PointStruct(id=id, vector=embedding, payload=metadata)]
                            )
                    else:
                        # Batch new embeddings
                        full_text = self.truncate_text(title, metadata['description'], text)
                        current_batch.append(full_text)
                        batch_metadata.append((id, metadata))
                        
                        if len(current_batch) >= batch_size:
                            embeddings = self.batch_embed_texts(current_batch, embed_model)
                            points = [
                                models.PointStruct(id=meta[0], vector=emb, payload=meta[1])
                                for emb, meta in zip(embeddings, batch_metadata)
                            ]
                            self.client.upsert(collection_name=collection_name, points=points)
                            current_batch = []
                            batch_metadata = []
                    
                    pbar.update(1)
            
            # Process any remaining items in the batch
            if current_batch:
                embeddings = self.batch_embed_texts(current_batch, embed_model)
                points = [
                    models.PointStruct(id=meta[0], vector=emb, payload=meta[1])
                    for emb, meta in zip(embeddings, batch_metadata)
                ]
                self.client.upsert(collection_name=collection_name, points=points)

def main():
    fire.Fire(PostEmbedder)

if __name__ == "__main__":
    main()