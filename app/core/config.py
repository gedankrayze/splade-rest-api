"""
Application configuration
"""

from enum import Enum

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class FAISSIndexType(str, Enum):
    """Enum for FAISS index types"""
    FLAT = "flat"  # IndexFlatIP - Exact search, most accurate but slowest
    IVF = "ivf"  # IndexIVFFlat - Approximate search with inverted file structure
    HNSW = "hnsw"  # IndexHNSWFlat - Hierarchical Navigable Small World graph


class Settings(BaseSettings):
    """Application settings"""

    model_config = ConfigDict(
        env_file=".env",
        env_prefix="SPLADE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SPLADE Content Server"
    PORT: int = 3000

    # LLM integration settings
    LLM_DEFAULT_MODEL: str = "gpt-4o-mini"
    LLM_ENABLED: bool = True

    # SPLADE model settings
    MODEL_NAME: str = "splade-cocondenser-selfdistil"  # Name for the model directory
    MODEL_DIR: str = "./models/{model_name}"  # Template with {model_name} placeholder
    MODEL_HF_ID: str = "baseplate/splade-cocondenser-selfdistil"  # Hugging Face model ID to download if MODEL_DIR is empty
    AUTO_DOWNLOAD_MODEL: bool = True  # Whether to automatically download the model if not found
    MAX_LENGTH: int = 512

    # Data storage settings
    DATA_DIR: str = "app/data"

    # Search settings
    DEFAULT_TOP_K: int = 10
    MIN_SCORE_THRESHOLD: float = 0.3  # Minimum similarity score for results

    # Hybrid dense reranking settings
    DENSE_RERANK_ENABLED: bool = True
    DENSE_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    HYBRID_CANDIDATE_MULTIPLIER: int = 20
    HYBRID_MAX_CANDIDATES: int = 200

    # Chunking settings
    MAX_CHUNK_SIZE: int = 500  # Maximum tokens per regular chunk
    TABLE_CHUNK_SIZE: int = 1000  # Maximum tokens for table chunks
    CHUNK_OVERLAP: int = 50  # Overlap tokens between chunks

    # Geo-spatial settings
    GEO_INDEX_PRECISION: int = 6  # Grid precision for spatial index
    GEO_DEFAULT_RADIUS_KM: float = 10.0  # Default search radius

    # FAISS index settings
    FAISS_INDEX_TYPE: FAISSIndexType = FAISSIndexType.FLAT  # Default to exact search
    FAISS_NLIST: int = 100  # Number of clusters for IVF indexes
    FAISS_HNSW_M: int = 32  # Number of connections for HNSW graph
    FAISS_SEARCH_NPROBE: int = 10  # Number of clusters to search for IVF

    # Document update settings
    SOFT_DELETE_ENABLED: bool = True  # Enable soft deletion of documents
    INDEX_REBUILD_THRESHOLD: int = 100  # Rebuild index after this many deletions


settings = Settings()
