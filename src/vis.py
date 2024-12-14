"""
Visualize post embeddings using UMAP with interactive hover.

Example usage:
python vis.py visualize_embeddings \
    --collection-name="post-collection-v1" \
    --n-neighbors=15 \
    --min-dist=0.1
"""
import fire
import os
from dotenv import load_dotenv
import numpy as np
from qdrant_client import QdrantClient
import plotly.express as px
import pandas as pd
from umap import UMAP

class EmbeddingVisualizer:
    def __init__(self):
        load_dotenv()
        self.client = QdrantClient(
            os.environ["QDRANT_URI"],
            api_key=os.environ["QDRANT_API_KEY"]
        )

    def visualize_embeddings(
        self,
        collection_name: str,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ):
        """
        Create interactive UMAP visualization of post embeddings.
        
        Args:
            collection_name: Name of Qdrant collection containing embeddings
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points for UMAP
        """
        # Get all points from collection
        points = self.client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust if needed
            with_vectors=True,
            with_payload=True
        )[0]

        # Extract embeddings and metadata
        embeddings = np.array([p.vector for p in points])
        metadata = [p.payload for p in points]

        # Create DataFrame with metadata
        df = pd.DataFrame(metadata)
        
        # Generate UMAP projection
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embedding = reducer.fit_transform(embeddings)
        
        # Add UMAP coordinates to DataFrame
        df['UMAP1'] = embedding[:, 0]
        df['UMAP2'] = embedding[:, 1]

        # Create hover text
        df['hover_text'] = df.apply(
            lambda x: f"Title: {x['title']}<br>"
                     f"Description: {x['description']}<br>"
                     f"URL: {x['canonical_url']}", 
            axis=1
        )

        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            hover_data=['title', 'description', 'canonical_url'],
            title='UMAP Projection of Post Embeddings'
        )

        fig.update_traces(
            marker=dict(size=5),
            hovertemplate="<br>".join([
                "Title: %{customdata[0]}",
                "Description: %{customdata[1]}",
                "URL: %{customdata[2]}",
                "<extra></extra>"
            ])
        )

        fig.show()

def main():
    fire.Fire(EmbeddingVisualizer)

if __name__ == "__main__":
    main()
