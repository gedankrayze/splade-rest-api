"""
FAISS index factory for creating and managing different index types
"""

import logging
from typing import Optional

import faiss
import numpy as np

from app.core.config import settings, FAISSIndexType

# Configure logging
logger = logging.getLogger("faiss_factory")


class FAISSIndexFactory:
    """Factory for creating different types of FAISS indexes"""

    @staticmethod
    def create_index(vocab_size: int, vectors: Optional[np.ndarray] = None) -> faiss.Index:
        """
        Create a FAISS index based on configuration settings
        
        Args:
            vocab_size: Vocabulary size (dimension of vectors)
            vectors: Optional training vectors for indexes that require training
            
        Returns:
            A FAISS index of the configured type
        """
        index_type = settings.FAISS_INDEX_TYPE

        logger.info(f"Creating FAISS index of type: {index_type}")

        if index_type == FAISSIndexType.FLAT:
            # Simple exact search index - no training needed
            index = faiss.IndexFlatIP(vocab_size)
            logger.info(f"Created IndexFlatIP with dimension {vocab_size}")
            return index

        elif index_type == FAISSIndexType.IVF:
            # IVF requires training data
            if vectors is None or len(vectors) < settings.FAISS_NLIST * 10:
                logger.warning("Not enough training vectors for IVF index, falling back to IndexFlatIP")
                return faiss.IndexFlatIP(vocab_size)

            # Create the IVF index
            nlist = min(settings.FAISS_NLIST, len(vectors) // 10)  # Ensure reasonable nlist
            quantizer = faiss.IndexFlatIP(vocab_size)
            index = faiss.IndexIVFFlat(quantizer, vocab_size, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index
            logger.info(f"Training IndexIVFFlat with {len(vectors)} vectors, nlist={nlist}")
            index.train(vectors)

            # Set search parameters
            index.nprobe = settings.FAISS_SEARCH_NPROBE

            logger.info(f"Created IndexIVFFlat with dimension {vocab_size}, nlist={nlist}, nprobe={index.nprobe}")
            return index

        elif index_type == FAISSIndexType.HNSW:
            # Create HNSW index
            M = settings.FAISS_HNSW_M

            # HNSW with inner product is a bit tricky in FAISS
            # We need to use an alternative implementation
            try:
                # Try to create native inner product version
                index = faiss.IndexHNSWFlat(vocab_size, M, faiss.METRIC_INNER_PRODUCT)
                logger.info(f"Created IndexHNSWFlat with dimension {vocab_size}, M={M}")
            except Exception as e:
                logger.warning(f"Error creating native HNSW index: {e}, using alternative")
                # Alternative: Create L2 index and convert vectors (less efficient but more compatible)
                index = faiss.IndexHNSWFlat(vocab_size, M)
                logger.info(f"Created alternative IndexHNSWFlat with dimension {vocab_size}, M={M}")

            return index

        # Fallback to flat index if type is invalid
        logger.warning(f"Unknown index type: {index_type}, falling back to IndexFlatIP")
        return faiss.IndexFlatIP(vocab_size)

    @staticmethod
    def convert_to_index_type(old_index: faiss.Index, target_type: FAISSIndexType, vectors: np.ndarray) -> faiss.Index:
        """
        Convert an existing index to a different type, preserving the vectors
        
        Args:
            old_index: Existing FAISS index
            target_type: Target index type
            vectors: Vectors to add to the new index
            
        Returns:
            New FAISS index of the target type with the same vectors
        """
        if old_index.ntotal == 0:
            # If the index is empty, just create a new one
            return FAISSIndexFactory.create_index(old_index.d, vectors if vectors is not None else None)

        # If we have no vectors provided but need them, extract from old index if possible
        if vectors is None and target_type in [FAISSIndexType.IVF, FAISSIndexType.HNSW]:
            try:
                # Try to reconstruct vectors from the index
                vectors = np.zeros((old_index.ntotal, old_index.d), dtype=np.float32)
                for i in range(old_index.ntotal):
                    vectors[i] = old_index.reconstruct(i)
            except RuntimeError:
                logger.error("Cannot extract vectors from old index and no vectors provided")
                return old_index  # Return the old index unchanged

        # Create new index and add vectors
        new_index = FAISSIndexFactory.create_index(old_index.d, vectors)

        # Add vectors to the new index if we have them
        if vectors is not None and vectors.shape[0] > 0:
            # Normalize vectors for inner product (if not already normalized)
            faiss.normalize_L2(vectors)
            new_index.add(vectors)

        return new_index
