#!/usr/bin/env python3
"""
Test script for domain-specific models with collections
"""

import os
import sys
import time

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.splade_service import splade_service
from app.models.schema import Document
from app.core.config import settings


def test_multi_model_collections():
    """Test collections with different domain-specific models"""
    print("=" * 50)
    print("MULTI-MODEL COLLECTIONS TEST")
    print("=" * 50)
    
    # Get available models - for the test we'll use the same model with different names
    # In a real scenario, you would have different domain-specific models
    default_model_name = settings.MODEL_NAME
    domain_model_name = "test_domain_model"  # This will be a copy of the default model
    
    # Create a copy of the default model for testing
    default_model_dir = settings.MODEL_DIR.format(model_name=default_model_name)
    domain_model_dir = settings.MODEL_DIR.format(model_name=domain_model_name)
    
    # Check if default model exists
    if not os.path.exists(os.path.join(default_model_dir, "config.json")):
        print(f"Default model not found at {default_model_dir}")
        print("Please ensure the default model is downloaded before running this test")
        return
    
    # Create domain model directory if it doesn't exist
    if not os.path.exists(domain_model_dir):
        os.makedirs(domain_model_dir, exist_ok=True)
        
        # Create symlinks to the default model files for testing
        # This simulates having a different model without actually duplicating files
        for filename in os.listdir(default_model_dir):
            src = os.path.join(default_model_dir, filename)
            dst = os.path.join(domain_model_dir, filename)
            
            if os.path.isfile(src):
                # Create symlink
                try:
                    os.symlink(src, dst)
                    print(f"Created symlink: {dst} -> {src}")
                except Exception as e:
                    # If symlink fails, try to copy the file
                    print(f"Could not create symlink: {e}")
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                        print(f"Copied file instead: {src} -> {dst}")
                    except Exception as e2:
                        print(f"Could not copy file: {e2}")
                        return
    
    # Create test collections
    collections = [
        {
            "id": "default-model-collection",
            "name": "Default Model Collection",
            "description": "Collection using the default model",
            "model_name": None  # Will use default model
        },
        {
            "id": "domain-model-collection",
            "name": "Domain Model Collection",
            "description": "Collection using a domain-specific model",
            "model_name": domain_model_name
        }
    ]
    
    # Create collections
    for collection in collections:
        # Delete if exists
        if collection["id"] in [c["id"] for c in splade_service.list_collections()]:
            splade_service.delete_collection(collection["id"])
            
        # Create collection
        print(f"Creating collection: {collection['id']} with model: {collection['model_name'] or 'default'}")
        splade_service.create_collection(
            collection_id=collection["id"],
            name=collection["name"],
            description=collection["description"],
            model_name=collection["model_name"]
        )
    
    # Add test documents to each collection
    test_documents = [
        Document(
            id="doc1",
            content="This is a test document about artificial intelligence and machine learning.",
            metadata={"category": "AI"}
        ),
        Document(
            id="doc2",
            content="Natural language processing is a subfield of artificial intelligence.",
            metadata={"category": "NLP"}
        ),
        Document(
            id="doc3",
            content="Deep learning models have revolutionized computer vision applications.",
            metadata={"category": "CV"}
        )
    ]
    
    # Add documents to each collection
    for collection_id in [c["id"] for c in collections]:
        print(f"\nAdding documents to collection: {collection_id}")
        for doc in test_documents:
            # Create a copy with a unique ID
            doc_copy = Document(
                id=f"{collection_id}-{doc.id}",
                content=doc.content,
                metadata=doc.metadata
            )
            result = splade_service.add_document(collection_id, doc_copy, auto_chunk=False)
            print(f"  Added document {doc_copy.id}: {result}")
    
    # Test search with each collection
    query = "artificial intelligence"
    print(f"\nSearching for: '{query}' in each collection")
    
    for collection_id in [c["id"] for c in collections]:
        print(f"\nSearching in {collection_id}:")
        
        # Get collection details to show which model it's using
        collection = splade_service.get_collection(collection_id)
        model = collection.get("model_name") or "default"
        print(f"  Using model: {model}")
        
        # Perform search
        start_time = time.time()
        results, query_time = splade_service.search(
            collection_id, 
            query,
            top_k=5
        )
        total_time = (time.time() - start_time) * 1000
        
        # Display results
        print(f"  Found {len(results)} results in {query_time:.2f}ms (total: {total_time:.2f}ms)")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['id']} - Score: {result['score']:.4f}")
            print(f"     {result['content'][:50]}...")
    
    print("\nVerifying model caching...")
    loaded_models = list(splade_service.models.keys())
    print(f"Currently loaded models: {loaded_models}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_multi_model_collections()
