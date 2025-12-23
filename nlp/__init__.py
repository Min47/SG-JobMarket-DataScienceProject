"""NLP package.

Embeddings generation, language cleaning, and text normalization.

Modules:
- embeddings: Sentence-BERT embedding generation
- generate_embeddings: CLI for batch embedding generation to BigQuery

Usage:
    from nlp.embeddings import EmbeddingGenerator, embed_texts
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_texts(["Data Scientist role..."])
    
    # CLI usage:
    # python -m nlp.generate_embeddings --limit 1000
"""

from nlp.embeddings import EmbeddingGenerator, embed_texts, get_embedding_dimension

__all__ = [
    "EmbeddingGenerator",
    "embed_texts",
    "get_embedding_dimension",
]

