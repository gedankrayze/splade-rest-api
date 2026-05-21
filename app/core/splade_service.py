"""
Core SPLADE service with FAISS integration and collection support
"""

import logging
import os
import pickle
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from app.core.chunking import document_chunker
from app.core.config import settings, FAISSIndexType
from app.core.faiss_factory import FAISSIndexFactory
from app.core.geo_spatial_index import GeoSpatialIndex
from app.core.model_downloader import download_model_if_needed
from app.models.schema import Document, GeoCoordinates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("splade_service")


class SpladeCollection:
    """
    SPLADE Collection with FAISS index for a group of documents
    """

    def __init__(
            self,
            collection_id: str,
            name: str,
            description: Optional[str] = None,
            vocab_size: int = 30522,  # Default BERT vocab size
            data_dir: str = settings.DATA_DIR,
            index_type: FAISSIndexType = settings.FAISS_INDEX_TYPE,
            model_name: Optional[str] = None  # Optional specific model for this collection
    ):
        """Initialize a SPLADE collection with its own FAISS index"""
        self.id = collection_id
        self.name = name
        self.description = description
        self.vocab_size = vocab_size
        self.data_dir = data_dir
        self.index_type = index_type
        self.model_name = model_name  # Store the model name used for this collection

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Collection data paths
        self.index_path = os.path.join(data_dir, f"{collection_id}_index.faiss")
        self.data_path = os.path.join(data_dir, f"{collection_id}_data.pkl")

        # Storage for documents and mappings
        self.documents = {}  # id -> Document
        self.dense_vectors = {}  # id -> dense embedding vector
        self.id_to_index = {}  # id -> index in FAISS
        self.index_to_id = {}  # index in FAISS -> id
        self.next_index = 0  # Next index to use in FAISS

        # Soft deletion tracking
        self.soft_delete_enabled = settings.SOFT_DELETE_ENABLED
        self.deleted_ids = set()  # Set of soft-deleted document IDs
        self.deletion_count = 0  # Counter for tracking deletion threshold
        self.rebuild_threshold = settings.INDEX_REBUILD_THRESHOLD

        # Initialize FAISS index
        self.index = FAISSIndexFactory.create_index(vocab_size)

        # Initialize geo-spatial index
        self.geo_index = GeoSpatialIndex(precision=settings.GEO_INDEX_PRECISION)

        # Lock for thread safety
        self.lock = threading.RLock()

        # Load existing data if available
        self._load_data()

    def add_document(self, doc: Document, vector: np.ndarray, dense_vector: Optional[np.ndarray] = None) -> bool:
        """Add a document to the collection with its pre-encoded vector"""
        with self.lock:
            if doc.id in self.documents:
                logger.warning(f"Document with ID {doc.id} already exists in collection {self.id}")
                return False

            # Normalize vector for cosine similarity
            faiss.normalize_L2(vector.reshape(1, -1))

            # Add to FAISS index
            self.index.add(vector.reshape(1, -1))

            # Add to geo-spatial index if location is present
            if doc.location:
                self.geo_index.add_document(
                    doc.id,
                    doc.location.latitude,
                    doc.location.longitude
                )
                logger.debug(f"Added document {doc.id} to geo-spatial index at "
                             f"({doc.location.latitude}, {doc.location.longitude})")

            # Store document and mapping
            self.documents[doc.id] = doc
            if dense_vector is not None:
                self.dense_vectors[doc.id] = dense_vector.astype(np.float32)
            self.id_to_index[doc.id] = self.next_index
            self.index_to_id[self.next_index] = doc.id
            self.next_index += 1

            # Persist changes
            self._save_data()

            logger.info(f"Added document {doc.id} to collection {self.id}")
            return True

    def batch_add_documents(
            self,
            docs: List[Document],
            vectors: List[np.ndarray],
            dense_vectors: Optional[List[np.ndarray]] = None
    ) -> int:
        """Add multiple documents to the collection in batch"""
        with self.lock:
            if len(docs) != len(vectors):
                raise ValueError("Number of documents and vectors must match")

            if dense_vectors is not None and len(docs) != len(dense_vectors):
                raise ValueError("Number of documents and dense vectors must match")

            # Filter out documents that already exist
            new_docs = []
            new_vectors = []
            new_dense_vectors = []
            for i, (doc, vec) in enumerate(zip(docs, vectors)):
                if doc.id not in self.documents:
                    new_docs.append(doc)
                    new_vectors.append(vec)
                    if dense_vectors is not None:
                        new_dense_vectors.append(dense_vectors[i])

            if not new_docs:
                logger.warning(f"No new documents to add to collection {self.id}")
                return 0

            # Normalize vectors
            vectors_array = np.vstack(new_vectors)
            faiss.normalize_L2(vectors_array)

            # Add to FAISS index
            self.index.add(vectors_array)

            # Store documents and mappings
            for i, doc in enumerate(new_docs):
                self.documents[doc.id] = doc
                if dense_vectors is not None:
                    self.dense_vectors[doc.id] = new_dense_vectors[i].astype(np.float32)
                self.id_to_index[doc.id] = self.next_index
                self.index_to_id[self.next_index] = doc.id
                self.next_index += 1

                if doc.location:
                    self.geo_index.add_document(
                        doc.id,
                        doc.location.latitude,
                        doc.location.longitude
                    )

            # Persist changes
            self._save_data()

            logger.info(f"Added {len(new_docs)} documents to collection {self.id}")
            return len(new_docs)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the collection"""
        with self.lock:
            if doc_id not in self.documents:
                logger.warning(f"Document with ID {doc_id} not found in collection {self.id}")
                return False

            # Check if we should use soft deletion
            if self.soft_delete_enabled:
                # Mark document as deleted but keep it in the index
                logger.info(f"Soft-deleting document {doc_id} from collection {self.id}")
                self.deleted_ids.add(doc_id)
                self.deletion_count += 1

                # Save the updated deletion tracking
                self._save_data()

                # Check if we've reached the threshold for rebuilding the index
                if self.deletion_count >= self.rebuild_threshold:
                    logger.info(f"Deletion threshold reached ({self.deletion_count} deletions), "
                                f"scheduling index rebuild for collection {self.id}")
                    return "rebuild"  # Signal the parent service to rebuild

                return True

            # If soft deletion is disabled, proceed with standard removal
            logger.info(f"Removing document {doc_id} from collection {self.id}")

            # Get index in FAISS
            removed_index = self.id_to_index[doc_id]

            # Remove from geo-spatial index if applicable
            self.geo_index.remove_document(doc_id)

            # Remove from storage
            del self.documents[doc_id]
            self.dense_vectors.pop(doc_id, None)
            del self.id_to_index[doc_id]
            del self.index_to_id[removed_index]

            # If collection is now empty, just reset the index
            if not self.documents:
                self.index = FAISSIndexFactory.create_index(self.vocab_size)
                self.next_index = 0
                self._save_data()
                return True

            # Otherwise, we need to rebuild the index
            # This is done by the parent service which has access to the model
            return "rebuild"  # Signal the parent service to rebuild

    def search(self, query_vector: np.ndarray, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None,
               geo_filter: Optional[Dict[str, Any]] = None,
               min_score: float = settings.MIN_SCORE_THRESHOLD) -> List[Dict[str, Any]]:
        """Search the collection with a query vector"""
        with self.lock:
            # If collection is empty, return empty results
            if self.index.ntotal == 0:
                return []

            # Process geo filter if provided
            geo_filtered_ids = None
            if geo_filter and "latitude" in geo_filter and "longitude" in geo_filter:
                lat = geo_filter["latitude"]
                lon = geo_filter["longitude"]
                radius_km = geo_filter.get("radius_km", settings.GEO_DEFAULT_RADIUS_KM)

                # Get IDs within radius
                geo_filtered_ids = self.geo_index.search_radius(lat, lon, radius_km)

                # If no documents match geo filter, return empty results
                if not geo_filtered_ids:
                    logger.debug(f"No documents found within {radius_km}km of ({lat}, {lon})")
                    return []

                logger.debug(
                    f"Geo filter found {len(geo_filtered_ids)} documents within {radius_km}km of ({lat}, {lon})")

            # Normalize query vector
            query_vector_normalized = query_vector.copy()
            faiss.normalize_L2(query_vector_normalized.reshape(1, -1))

            # For approximate indexes like IVF, make sure search parameters are set
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = settings.FAISS_SEARCH_NPROBE

            # Search FAISS index - get more results than needed to account for filtering
            # Get even more if we have soft-deleted documents or geo filtering
            extra_results = 3
            if self.soft_delete_enabled and self.deleted_ids:
                extra_results = max(5, len(self.deleted_ids) // 10)  # Adjust based on deletion count

            if geo_filtered_ids:
                # If geo filtering active, we need more results to ensure we have enough after filtering
                extra_results = max(extra_results, len(geo_filtered_ids) // 5)
                
            scores, indices = self.index.search(
                query_vector_normalized.reshape(1, -1),
                min(top_k * extra_results, self.index.ntotal)  # Get extra results for filtering
            )

            results = []
            for i, idx in enumerate(indices[0]):
                # Skip invalid indices or scores below threshold
                if idx == -1 or idx not in self.index_to_id or scores[0][i] < min_score:  
                    continue

                doc_id = self.index_to_id[idx]

                # Skip soft-deleted documents
                if self.soft_delete_enabled and doc_id in self.deleted_ids:
                    continue

                # Skip documents not matching geo filter
                if geo_filtered_ids is not None and doc_id not in geo_filtered_ids:
                    continue
                    
                doc = self.documents[doc_id]

                # Apply metadata filtering if specified
                if filter_metadata and doc.metadata:
                    skip = False
                    for key, value in filter_metadata.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue

                # Prepare result with basic info
                result = {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata or {},
                    "score": float(scores[0][i])
                }

                # Add location to result if available
                if doc.location:
                    result["location"] = {
                        "latitude": doc.location.latitude,
                        "longitude": doc.location.longitude
                    }

                    # If geo search, calculate and add distance
                    if geo_filter:
                        coords = self.geo_index.get_location(doc_id)
                        if coords:
                            distance = self.geo_index._haversine_distance(
                                geo_filter["latitude"],
                                geo_filter["longitude"],
                                coords[0], coords[1]
                            )
                            result["distance_km"] = round(distance, 2)

                results.append(result)

                # If we have enough results after filtering, stop
                if len(results) >= top_k:
                    break

            return results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        with self.lock:
            return self.documents.get(doc_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "document_count": len(self.documents),
                "index_size": self.index.ntotal,
                "vocab_size": self.vocab_size
            }

    def _save_data(self):
        """Save collection data to disk"""
        with self.lock:
            # Save document store
            with open(self.data_path, "wb") as f:
                pickle.dump({
                    "id": self.id,
                    "name": self.name,
                    "description": self.description,
                    "documents": self.documents,
                    "dense_vectors": self.dense_vectors,
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                    "next_index": self.next_index,
                    "index_type": self.index_type,
                    "deleted_ids": self.deleted_ids,
                    "deletion_count": self.deletion_count,
                    "model_name": self.model_name,  # Save the model name
                    # We don't save the geo_index directly, as it will be reconstructed from document locations
                }, f)

            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            doc_count = len(self.documents)
            deleted_count = len(self.deleted_ids) if self.soft_delete_enabled else 0
            geo_count = len(self.geo_index.doc_to_coords)
            logger.info(f"Collection {self.id} persisted to disk: {doc_count} documents "
                        f"({deleted_count} soft-deleted, {geo_count} with location)")

    def _load_data(self):
        """Load collection data from disk"""
        try:
            # Check if data files exist
            if not (os.path.exists(self.data_path) and os.path.exists(self.index_path)):
                logger.info(f"No existing data found for collection {self.id}, starting with empty collection")
                return

            # Load document store
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                self.name = data["name"]
                self.description = data["description"]
                self.documents = data["documents"]
                self.dense_vectors = data.get("dense_vectors", {})
                self.id_to_index = data["id_to_index"]
                self.index_to_id = data["index_to_id"]
                self.next_index = data["next_index"]

                # Load soft deletion tracking if available
                if "deleted_ids" in data:
                    self.deleted_ids = data["deleted_ids"]
                else:
                    self.deleted_ids = set()

                if "deletion_count" in data:
                    self.deletion_count = data["deletion_count"]
                else:
                    self.deletion_count = 0

                # Check if we need to migrate index type
                if "index_type" in data:
                    self.index_type = data["index_type"]

                    # If configured index type is different from saved one,
                    # we'll convert it after loading
                    needs_conversion = self.index_type != settings.FAISS_INDEX_TYPE

                # Load model name if available
                if "model_name" in data:
                    self.model_name = data["model_name"]

            # Load FAISS index
            self.index = faiss.read_index(self.index_path)

            # Convert index type if needed and if setting has changed
            if 'needs_conversion' in locals() and needs_conversion:
                logger.info(f"Converting index from {self.index_type} to {settings.FAISS_INDEX_TYPE}")
                # We would need to reconstruct vectors for conversion
                # This is complex and left as a future enhancement
                self.index_type = settings.FAISS_INDEX_TYPE

            # Rebuild geo-spatial index from document locations
            self.geo_index = GeoSpatialIndex(precision=settings.GEO_INDEX_PRECISION)
            geo_count = 0

            for doc_id, doc in self.documents.items():
                # Skip documents marked as deleted
                if self.soft_delete_enabled and doc_id in self.deleted_ids:
                    continue

                # Add locations to geo index
                if hasattr(doc, 'location') and doc.location:
                    self.geo_index.add_document(
                        doc_id,
                        doc.location.latitude,
                        doc.location.longitude
                    )
                    geo_count += 1
                # Legacy: check if location info is in metadata
                elif doc.metadata and 'latitude' in doc.metadata and 'longitude' in doc.metadata:
                    lat = doc.metadata['latitude']
                    lon = doc.metadata['longitude']
                    self.geo_index.add_document(doc_id, lat, lon)

                    # Upgrade to new location field model
                    doc.location = GeoCoordinates(latitude=lat, longitude=lon)
                    geo_count += 1

            doc_count = len(self.documents)
            deleted_count = len(self.deleted_ids) if self.soft_delete_enabled else 0
            logger.info(f"Loaded collection {self.id} with {doc_count} documents "
                        f"({deleted_count} soft-deleted, {geo_count} with location)")
        except Exception as e:
            logger.error(f"Error loading data for collection {self.id}: {e}")
            logger.info(f"Starting with empty collection for {self.id}")


class SpladeService:
    """
    SPLADE service with collection management
    """

    def __init__(self, model_name: str = settings.MODEL_NAME, data_dir: str = settings.DATA_DIR):
        """Initialize SPLADE service"""
        # Format the model directory template with the actual model name
        self.default_model_name = model_name
        self.data_dir = data_dir

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Path for collections metadata
        self.collections_path = os.path.join(data_dir, "collections.pkl")

        # Model cache to avoid loading the same model multiple times
        self.models = {}
        self.tokenizers = {}
        self.dense_embedding_model = None

        # Load the default model
        self._load_model(model_name)

        # Get vocabulary size from the default model
        self.vocab_size = self.models[model_name].config.vocab_size
        logger.info(f"Default model vocabulary size: {self.vocab_size}")

        # Collection storage
        self.collections = {}  # id -> SpladeCollection

        # Lock for thread safety
        self.lock = threading.RLock()

        # Load existing collections
        self._load_collections()

    def _load_model(self, model_name: str) -> bool:
        """Load a specific model and add to cache"""
        if model_name in self.models:
            # Model already loaded
            return True

        # Format the model directory
        model_dir = settings.MODEL_DIR.format(model_name=model_name)

        # Check if model exists, download if not and if auto-download is enabled
        if settings.AUTO_DOWNLOAD_MODEL and not os.path.exists(os.path.join(model_dir, "config.json")):
            logger.info(f"Model not found at {model_dir}, attempting to download...")
            download_success = download_model_if_needed(model_dir, settings.MODEL_HF_ID)
            if not download_success:
                logger.error(f"Failed to download model {settings.MODEL_HF_ID} to {model_dir}")
                return False

        try:
            # Initialize tokenizer and model
            logger.info(f"Loading model from {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForMaskedLM.from_pretrained(model_dir, use_safetensors=True)

            # Set device
            if torch.backends.mps.is_available():
                logger.info(f"Using MPS for model {model_name}")
                device = torch.device("mps")
            elif torch.cuda.is_available():
                logger.info(f"Using CUDA for model {model_name}")
                device = torch.device("cuda")
            else:
                logger.info(f"Using CPU for model {model_name}")
                device = torch.device("cpu")

            model.to(device)
            model.eval()

            # Store in cache
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def encode_text(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """
        Encode text into SPLADE sparse representation
        
        Args:
            text: Text to encode
            model_name: Optional model name to use (defaults to collection's model or system default)
        """
        # Use the specified model or default
        model_to_use = model_name or self.default_model_name

        # Check if model is loaded
        if model_to_use not in self.models:
            # Try to load the model
            if not self._load_model(model_to_use):
                # If loading fails, fall back to the default model
                logger.warning(
                    f"Could not load model {model_to_use}, falling back to default {self.default_model_name}")
                model_to_use = self.default_model_name

        # Get the appropriate model and tokenizer
        model = self.models[model_to_use]
        tokenizer = self.tokenizers[model_to_use]

        # Determine the device for this model
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Ensure model is on the right device
        model.to(device)

        # Tokenize the input
        inputs = tokenizer(
            text,
            max_length=settings.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate the sparse representation
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        # Apply SPLADE pooling (log(1 + ReLU(x)))
        logits = outputs.logits
        activated = torch.log(1 + torch.relu(logits))

        # Max pooling over sequence dimension
        attention_expanded = inputs["attention_mask"].unsqueeze(-1).expand_as(activated)
        masked_activated = activated * attention_expanded
        sparse_rep = torch.max(masked_activated, dim=1)[0]

        return sparse_rep.cpu().numpy()[0]

    def batch_encode_texts(self, texts: List[str], model_name: Optional[str] = None) -> List[np.ndarray]:
        """
        Encode multiple texts in batches
        
        Args:
            texts: List of texts to encode
            model_name: Optional model name to use
        """
        results = []
        for text in texts:
            vector = self.encode_text(text, model_name)
            results.append(vector)
        return results

    def _load_dense_embedding_model(self):
        """Load the configured fastembed model lazily."""
        if self.dense_embedding_model is None:
            from fastembed import TextEmbedding

            logger.info(f"Loading dense embedding model: {settings.DENSE_EMBEDDING_MODEL}")
            self.dense_embedding_model = TextEmbedding(model_name=settings.DENSE_EMBEDDING_MODEL)

        return self.dense_embedding_model

    def encode_dense_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode texts with fastembed for dense reranking."""
        if not texts or not settings.DENSE_RERANK_ENABLED:
            return []

        try:
            model = self._load_dense_embedding_model()
            vectors = [np.asarray(vector, dtype=np.float32) for vector in model.embed(texts)]
            return [self._normalize_dense_vector(vector) for vector in vectors]
        except Exception as e:
            logger.error(f"Dense embedding failed, continuing with sparse results only: {e}")
            return []

    @staticmethod
    def _normalize_dense_vector(vector: np.ndarray) -> np.ndarray:
        """Normalize dense vectors so dot product is cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _hybrid_candidate_count(top_k: int) -> int:
        """Number of sparse candidates to fetch before dense reranking."""
        return min(
            settings.HYBRID_MAX_CANDIDATES,
            max(top_k, top_k * settings.HYBRID_CANDIDATE_MULTIPLIER),
        )

    def _ensure_dense_vectors(self, collection: SpladeCollection, results: List[Dict[str, Any]]) -> None:
        """Backfill dense vectors for candidate documents missing persisted embeddings."""
        missing_docs = [
            collection.documents[result["id"]]
            for result in results
            if result["id"] in collection.documents and result["id"] not in collection.dense_vectors
        ]

        if not missing_docs:
            return

        dense_vectors = self.encode_dense_texts([doc.content for doc in missing_docs])
        if len(dense_vectors) != len(missing_docs):
            return

        for doc, vector in zip(missing_docs, dense_vectors):
            collection.dense_vectors[doc.id] = vector

        collection._save_data()

    def _dense_rerank(
            self,
            collection: SpladeCollection,
            query: str,
            results: List[Dict[str, Any]],
            top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank sparse candidates with dense cosine similarity."""
        if not settings.DENSE_RERANK_ENABLED or not results:
            return results[:top_k]

        self._ensure_dense_vectors(collection, results)

        query_vectors = self.encode_dense_texts([query])
        if not query_vectors:
            return results[:top_k]

        query_vector = query_vectors[0]
        reranked = []

        for result in results:
            dense_vector = collection.dense_vectors.get(result["id"])
            if dense_vector is None:
                continue

            dense_score = float(np.dot(query_vector, dense_vector))
            reranked_result = result.copy()
            reranked_result["sparse_score"] = result["score"]
            reranked_result["dense_score"] = dense_score
            reranked_result["score"] = dense_score
            reranked.append(reranked_result)

        if not reranked:
            return results[:top_k]

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]

    def create_collection(self, collection_id: str, name: str, description: Optional[str] = None,
                          model_name: Optional[str] = None) -> bool:
        """
        Create a new collection
        
        Args:
            collection_id: Unique ID for the collection
            name: Display name for the collection
            description: Optional description
            model_name: Optional domain-specific model to use for this collection
        """
        with self.lock:
            if collection_id in self.collections:
                logger.warning(f"Collection with ID {collection_id} already exists")
                return False

            # If a specific model is requested, try to load it
            if model_name and model_name not in self.models:
                if not self._load_model(model_name):
                    logger.warning(f"Could not load model {model_name} for collection {collection_id}, "
                                   f"using default model {self.default_model_name}")
                    model_name = None  # Reset to use default

            # Create new collection
            collection = SpladeCollection(
                collection_id=collection_id,
                name=name,
                description=description,
                vocab_size=self.vocab_size,
                data_dir=self.data_dir,
                model_name=model_name
            )

            # Add to collections
            self.collections[collection_id] = collection

            # Save collections metadata
            self._save_collections()

            logger.info(f"Created collection {collection_id}" +
                        (f" with model {model_name}" if model_name else ""))
            return True

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection"""
        with self.lock:
            if collection_id not in self.collections:
                logger.warning(f"Collection with ID {collection_id} not found")
                return False

            # Get collection
            collection = self.collections[collection_id]

            # Remove collection files
            try:
                if os.path.exists(collection.index_path):
                    os.remove(collection.index_path)
                if os.path.exists(collection.data_path):
                    os.remove(collection.data_path)
            except Exception as e:
                logger.error(f"Error removing collection files: {e}")

            # Remove from collections
            del self.collections[collection_id]

            # Save collections metadata
            self._save_collections()

            logger.info(f"Deleted collection {collection_id}")
            return True

    def add_document(self, collection_id: str, doc: Document, auto_chunk: bool = True) -> bool:
        """Add a document to a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return False

        # Get the model name for this collection
        collection = self.collections[collection_id]
        model_name = collection.model_name

        # If auto_chunk is enabled and document is large, split it into chunks
        if auto_chunk:
            # Get document chunks
            chunks = document_chunker.chunk_document(doc)

            if len(chunks) > 1:
                logger.info(f"Document {doc.id} split into {len(chunks)} chunks")

                # Add each chunk
                success = True
                for chunk in chunks:
                    # Encode document with the appropriate model
                    vector = self.encode_text(chunk.content, model_name)
                    dense_vectors = self.encode_dense_texts([chunk.content])
                    dense_vector = dense_vectors[0] if dense_vectors else None

                    # Add to collection (don't auto-chunk to avoid infinite recursion)
                    chunk_success = self.collections[collection_id].add_document(chunk, vector, dense_vector)
                    success = success and chunk_success

                return success

        # If not auto-chunking or document is small enough
        # Encode document with the appropriate model
        vector = self.encode_text(doc.content, model_name)
        dense_vectors = self.encode_dense_texts([doc.content])
        dense_vector = dense_vectors[0] if dense_vectors else None

        # Add to collection
        return self.collections[collection_id].add_document(doc, vector, dense_vector)

    def batch_add_documents(self, collection_id: str, docs: List[Document], auto_chunk: bool = True) -> int:
        """Add multiple documents to a collection in batch"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return 0

        # Get the model name for this collection
        collection = self.collections[collection_id]
        model_name = collection.model_name

        added_count = 0

        # If auto-chunking is enabled
        if auto_chunk:
            # Process each document, potentially splitting into chunks
            all_chunks = []
            for doc in docs:
                chunks = document_chunker.chunk_document(doc)
                all_chunks.extend(chunks)

            if len(all_chunks) > len(docs):
                logger.info(f"Split {len(docs)} documents into {len(all_chunks)} chunks")

                # Encode all chunks with the appropriate model
                vectors = self.batch_encode_texts([chunk.content for chunk in all_chunks], model_name)
                dense_vectors = self.encode_dense_texts([chunk.content for chunk in all_chunks])

                # Add chunks to collection
                added_count = self.collections[collection_id].batch_add_documents(
                    all_chunks,
                    vectors,
                    dense_vectors if len(dense_vectors) == len(all_chunks) else None
                )
                return added_count

        # If not auto-chunking or no documents were chunked
        # Encode documents with the appropriate model
        vectors = self.batch_encode_texts([doc.content for doc in docs], model_name)
        dense_vectors = self.encode_dense_texts([doc.content for doc in docs])

        # Add to collection
        return self.collections[collection_id].batch_add_documents(
            docs,
            vectors,
            dense_vectors if len(dense_vectors) == len(docs) else None
        )

    def remove_document(self, collection_id: str, doc_id: str) -> bool:
        """Remove a document from a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return False

        # Remove from collection
        result = self.collections[collection_id].remove_document(doc_id)

        # Handle the result from remove_document
        if result == "rebuild":
            # If the collection signals that we need to rebuild the index
            logger.info(f"Rebuilding index for collection {collection_id} as requested")
            self._rebuild_collection_index(collection_id)
            return True
        elif result and not self.collections[collection_id].soft_delete_enabled:
            # For non-soft deletion, always rebuild
            self._rebuild_collection_index(collection_id)

        return bool(result)  # Convert any truthy value to actual boolean

    def _rebuild_collection_index(self, collection_id: str):
        """Rebuild the FAISS index for a collection after document removal"""
        with self.lock:
            collection = self.collections[collection_id]

            # Get the model name for this collection
            model_name = collection.model_name

            # Get active documents (excluding soft-deleted ones)
            if collection.soft_delete_enabled and collection.deleted_ids:
                docs = [doc for doc_id, doc in collection.documents.items()
                        if doc_id not in collection.deleted_ids]

                # Also clean up the deleted IDs list and reset deletion counter
                for doc_id in list(collection.deleted_ids):
                    if doc_id not in collection.documents:
                        collection.deleted_ids.remove(doc_id)
                collection.deletion_count = 0

                logger.info(f"Rebuilding index for collection {collection_id}, "
                            f"excluding {len(collection.deleted_ids)} soft-deleted documents")
            else:
                docs = list(collection.documents.values())

            # Encode all documents with collection-specific model
            vectors = self.batch_encode_texts([doc.content for doc in docs], model_name)
            dense_vectors = self.encode_dense_texts([doc.content for doc in docs])

            # Create new mappings
            id_to_index = {}
            index_to_id = {}

            # Create a new index using the factory
            if vectors:
                vectors_array = np.vstack(vectors)
                faiss.normalize_L2(vectors_array)

                # Create appropriate index type with training data
                collection.index = FAISSIndexFactory.create_index(
                    self.vocab_size,
                    vectors_array
                )

                # Add vectors to the index
                collection.index.add(vectors_array)
            else:
                # Empty collection
                collection.index = FAISSIndexFactory.create_index(self.vocab_size)

            # Update mappings
            for i, doc in enumerate(docs):
                id_to_index[doc.id] = i
                index_to_id[i] = doc.id

            # Update collection
            collection.id_to_index = id_to_index
            collection.index_to_id = index_to_id
            collection.dense_vectors = {
                doc.id: dense_vectors[i]
                for i, doc in enumerate(docs)
            } if len(dense_vectors) == len(docs) else {}
            collection.next_index = len(docs)
            # Update index type to match what we created
            collection.index_type = settings.FAISS_INDEX_TYPE
            
            # Save collection
            collection._save_data()

            logger.info(f"Rebuilt index for collection {collection_id} with {len(docs)} documents")

    def search(self, collection_id: str, query: str, top_k: int = settings.DEFAULT_TOP_K,
               filter_metadata: Optional[Dict[str, Any]] = None,
               geo_filter: Optional[Dict[str, Any]] = None,
               min_score: float = settings.MIN_SCORE_THRESHOLD) -> \
            Tuple[List[Dict[str, Any]], float]:
        """Search for documents in a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return [], 0.0

        # Start timing
        start_time = time.time()

        # Get the model name for this collection
        collection = self.collections[collection_id]
        model_name = collection.model_name

        # Encode query using the appropriate model
        query_vector = self.encode_text(query, model_name)

        sparse_top_k = self._hybrid_candidate_count(top_k) if settings.DENSE_RERANK_ENABLED else top_k

        # Search collection
        results = self.collections[collection_id].search(
            query_vector, sparse_top_k, filter_metadata, geo_filter, min_score
        )
        results = self._dense_rerank(collection, query, results, top_k)

        # End timing
        query_time = (time.time() - start_time) * 1000  # in milliseconds

        return results, query_time

    def search_all_collections(self, query: str, top_k: int = settings.DEFAULT_TOP_K,
                               filter_metadata: Optional[Dict[str, Any]] = None,
                               geo_filter: Optional[Dict[str, Any]] = None,
                               min_score: float = settings.MIN_SCORE_THRESHOLD) -> Dict[str, Any]:
        """Search across all collections"""
        # Start timing
        start_time = time.time()

        # Search all collections
        all_results = {}
        for collection_id, collection in self.collections.items():
            # Get the model name for this collection
            model_name = collection.model_name

            # Encode query using the appropriate model
            query_vector = self.encode_text(query, model_name)

            sparse_top_k = self._hybrid_candidate_count(top_k) if settings.DENSE_RERANK_ENABLED else top_k

            # Search this collection
            collection_results = collection.search(
                query_vector, sparse_top_k, filter_metadata, geo_filter, min_score
            )
            collection_results = self._dense_rerank(collection, query, collection_results, top_k)
            if collection_results:
                all_results[collection_id] = collection_results

        # End timing
        query_time = (time.time() - start_time) * 1000  # in milliseconds

        return {
            "results": all_results,
            "query_time_ms": query_time
        }

    def get_document(self, collection_id: str, doc_id: str) -> Optional[Document]:
        """Get a document from a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return None

        # Get document from collection
        return self.collections[collection_id].get_document(doc_id)

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get collection details"""
        if collection_id not in self.collections:
            return None

        collection = self.collections[collection_id]
        return {
            "id": collection.id,
            "name": collection.name,
            "description": collection.description,
            "stats": collection.get_stats()
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        return [
            {
                "id": collection.id,
                "name": collection.name,
                "description": collection.description
            }
            for collection in self.collections.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        return {
            "collection_count": len(self.collections),
            "collections": [collection.get_stats() for collection in self.collections.values()]
        }

    def _save_collections(self):
        """Save collections metadata to disk"""
        with self.lock:
            collections_data = [
                {
                    "id": collection.id,
                    "name": collection.name,
                    "description": collection.description,
                    "model_name": collection.model_name  # Save the model name for each collection
                }
                for collection in self.collections.values()
            ]

            with open(self.collections_path, "wb") as f:
                pickle.dump(collections_data, f)

            logger.info(f"Saved metadata for {len(collections_data)} collections")

    def _load_collections(self):
        """Load collections from disk"""
        try:
            if not os.path.exists(self.collections_path):
                logger.info("No collections metadata found, starting with empty collections")
                return

            with open(self.collections_path, "rb") as f:
                collections_data = pickle.load(f)

            for collection_data in collections_data:
                collection_id = collection_data["id"]
                name = collection_data["name"]
                description = collection_data["description"]
                # Get model name if available, otherwise use None (will use default)
                model_name = collection_data.get("model_name")

                # Create collection object
                collection = SpladeCollection(
                    collection_id=collection_id,
                    name=name,
                    description=description,
                    vocab_size=self.vocab_size,
                    data_dir=self.data_dir,
                    model_name=model_name
                )

                self.collections[collection_id] = collection

                # Preload the model if it's not already loaded
                if model_name and model_name not in self.models:
                    try:
                        self._load_model(model_name)
                    except Exception as e:
                        logger.warning(f"Could not load model {model_name} for collection {collection_id}: {e}")

            logger.info(f"Loaded {len(self.collections)} collections")
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
            logger.info("Starting with empty collections")


class LazySpladeService:
    """Initialize the SPLADE service on first use instead of at import time."""

    def __init__(self):
        self._service: Optional[SpladeService] = None
        self._lock = threading.RLock()

    @property
    def is_initialized(self) -> bool:
        return self._service is not None

    def get_service(self) -> SpladeService:
        with self._lock:
            if self._service is None:
                self._service = SpladeService()
                logger.info(f"SPLADE service initialized with default model: {self._service.default_model_name}")
            return self._service

    def __getattr__(self, name: str):
        return getattr(self.get_service(), name)


# Initialize the SPLADE service lazily so imports, linting, and API startup do not
# download or load native ML dependencies until a search/indexing operation needs them.
splade_service = LazySpladeService()
