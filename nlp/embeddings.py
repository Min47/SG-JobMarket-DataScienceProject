"""
Embedding generation module for job descriptions.

Uses Sentence-BERT (all-MiniLM-L6-v2) to generate 384-dimensional embeddings.
Supports batched processing and BigQuery integration.

Usage:
    from nlp.embeddings import EmbeddingGenerator
    
    generator = EmbeddingGenerator()
    embeddings = generator.embed_texts(["Data Scientist role...", "Software Engineer..."])
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for job descriptions using Sentence-BERT."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast & good quality
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize embedding generator.

        Args:
            model_name: Sentence-BERT model name. Options:
                - "all-MiniLM-L6-v2" (384 dim, fast) ← default
                - "all-mpnet-base-v2" (768 dim, better quality)
        """
        self.model_name = model_name
        self._model = None  # Lazy load
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")

    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading Sentence-BERT model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: "
                    "pip install sentence-transformers"
                )
        return self._model

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.
            show_progress: Show tqdm progress bar.

        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        # Clean texts: handle None and empty strings
        cleaned_texts = [str(t) if t else "" for t in texts]

        logger.info(f"Embedding {len(cleaned_texts)} texts (batch_size={batch_size})")

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_job(
        self, job_title: str, job_description: str, max_desc_length: int = 1000
    ) -> np.ndarray:
        """
        Embed a single job by combining title and description.

        Args:
            job_title: Job title text.
            job_description: Job description text.
            max_desc_length: Truncate description to this length.

        Returns:
            1D numpy array of shape (EMBEDDING_DIM,)
        """
        # Combine title and truncated description
        desc = (job_description or "")[:max_desc_length]
        combined = f"{job_title or 'Unknown'}. {desc}"

        embedding = self.embed_texts([combined], show_progress=False)
        return embedding[0]


# Convenience function for backward compatibility
def embed_texts(texts: Iterable[str], model_name: str = EmbeddingGenerator.DEFAULT_MODEL) -> List[list[float]]:
    """
    Embed texts using Sentence-BERT.

    Args:
        texts: List of strings to embed.
        model_name: Model to use.

    Returns:
        List of embedding lists (for JSON serialization).
    """
    text_list = list(texts)
    if not text_list:
        return []
    generator = EmbeddingGenerator(model_name)
    embeddings = generator.embed_texts(text_list)
    return embeddings.tolist()


def get_embedding_dimension(model_name: str = EmbeddingGenerator.DEFAULT_MODEL) -> int:
    """Return the embedding dimension for a model."""
    dimensions = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "text-embedding-004": 768,  # Vertex AI
    }
    return dimensions.get(model_name, 384)


