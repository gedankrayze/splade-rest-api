#!/usr/bin/env python3
"""
Test script for FAISS index types and soft deletion
"""

import os
import sys
import time
import random

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.splade_service import splade_service
from app.models.schema import Document
from app.core.config import settings, FAISSIndexType


def test_faiss_index_types():
    """Test different FAISS index types"""
    # Create a test collection for each index type
    collections = {}
    
    for index_type in FAISSIndexType:
        # Update the global setting
        settings.FAISS_INDEX_TYPE = index_type
        
        collection_id = f"test-{index_type.value}"
        print(f"\nCreating collection for index type: {index_type.value}")
        
        # Create collection
        if collection_id in [c["id"] for c in splade_service.list_collections()]:
            # Delete existing collection
            splade_service.delete_collection(collection_id)
            
        splade_service.create_collection(
            collection_id=collection_id,
            name=f"Test {index_type.value.upper()} Index",
            description=f"Test collection for {index_type.value} index type"
        )
        
        collections[index_type] = collection_id
    
    # Add the same documents to each collection
    document_count = 50  # Use a small number for quick testing
    docs = []
    
    print(f"\nGenerating {document_count} test documents...")
    
    # Generate test documents
    for i in range(document_count):
        doc = Document(
            id=f"test-doc-{i}",
            content=f"This is test document {i} with some random words: " + 
                    " ".join(random.choice(["apple", "banana", "orange", "grape", "pear", 
                                            "search", "index", "vector", "database", "query",
                                            "document", "collection", "similarity", "retrieval", 
                                            "algorithm", "machine", "learning", "artificial", 
                                            "intelligence", "model", "embedding", "feature"])
                            for _ in range(20)),
            metadata={"index": i, "category": random.choice(["fruit", "tech", "random"])}
        )
        docs.append(doc)
    
    # Add documents to each collection
    for index_type, collection_id in collections.items():
        print(f"\nAdding documents to {index_type.value} collection...")
        start_time = time.time()
        
        # Update the global setting
        settings.FAISS_INDEX_TYPE = index_type
        
        # Add documents in batches
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            count = splade_service.batch_add_documents(collection_id, batch, auto_chunk=False)
            print(f"  Added batch of {count} documents")
        
        end_time = time.time()
        print(f"  Time to add {document_count} documents: {end_time - start_time:.3f} seconds")
    
    # Test search performance
    print("\nTesting search performance...")
    
    for index_type, collection_id in collections.items():
        # Update the global setting 
        settings.FAISS_INDEX_TYPE = index_type
        
        print(f"\nSearch test for {index_type.value}:")
        
        # Perform search and measure time
        start_time = time.time()
        results, query_time = splade_service.search(
            collection_id, 
            "search for similarity between documents", 
            top_k=5
        )
        end_time = time.time()
        
        # Print results
        print(f"  Found {len(results)} results in {query_time:.2f}ms")
        print(f"  Total time (including Python overhead): {(end_time - start_time) * 1000:.2f}ms")
        
        if results:
            print(f"  Top result: {results[0]['id']} (score: {results[0]['score']:.4f})")
            

def test_soft_deletion():
    """Test soft deletion of documents"""
    # Enable soft deletion
    original_setting = settings.SOFT_DELETE_ENABLED
    settings.SOFT_DELETE_ENABLED = True
    
    # Create a test collection
    collection_id = "test-soft-deletion"
    print("\nCreating collection for soft deletion test")
    
    # Create or clear collection
    if collection_id in [c["id"] for c in splade_service.list_collections()]:
        # Delete existing collection
        splade_service.delete_collection(collection_id)
        
    splade_service.create_collection(
        collection_id=collection_id,
        name="Test Soft Deletion",
        description="Test collection for soft deletion"
    )
    
    # Add documents
    document_count = 20
    docs = []
    
    # Generate test documents
    for i in range(document_count):
        doc = Document(
            id=f"deletion-doc-{i}",
            content=f"This is test document {i} for testing document deletion",
            metadata={"index": i}
        )
        docs.append(doc)
    
    # Add documents
    print(f"Adding {document_count} test documents...")
    splade_service.batch_add_documents(collection_id, docs, auto_chunk=False)
    
    # Perform a baseline search
    print("\nBaseline search before deletion:")
    query = "test document deletion"
    results, _ = splade_service.search(collection_id, query, top_k=10)
    print(f"  Found {len(results)} results")
    
    # Get list of documents to delete
    docs_to_delete = [doc.id for doc in docs[::4]]  # Delete every 4th document
    delete_count = len(docs_to_delete)
    
    print(f"\nDeleting {delete_count} documents (soft deletion) - IDs: {docs_to_delete}")
    
    # Delete documents one by one
    for doc_id in docs_to_delete:
        result = splade_service.remove_document(collection_id, doc_id)
        print(f"  Deleted {doc_id}: {result}")
    
    # Search again after deletion
    print("\nSearch after soft deletion:")
    results, _ = splade_service.search(collection_id, query, top_k=10)
    print(f"  Found {len(results)} results")
    
    # Verify deleted documents are not in results
    for result in results:
        if result["id"] in docs_to_delete:
            print(f"  ERROR: Deleted document {result['id']} still appears in results!")
    
    # Force a rebuild
    print("\nForcing index rebuild...")
    splade_service._rebuild_collection_index(collection_id)
    
    # Search again after rebuild
    print("\nSearch after rebuild:")
    results, _ = splade_service.search(collection_id, query, top_k=10)
    print(f"  Found {len(results)} results")
    
    # Update deletion threshold to trigger auto-rebuild
    original_threshold = settings.INDEX_REBUILD_THRESHOLD
    settings.INDEX_REBUILD_THRESHOLD = 2  # Set a low threshold
    
    print(f"\nSetting rebuild threshold to {settings.INDEX_REBUILD_THRESHOLD}")
    print("Deleting more documents to trigger automatic rebuild...")
    
    # Delete another batch of documents
    for i in range(3):  # Delete 3 more documents
        doc_id = f"deletion-doc-{i*4+1}"  # Delete 1, 5, 9
        result = splade_service.remove_document(collection_id, doc_id)
        print(f"  Deleted {doc_id}: {result}")
    
    # Search again after auto-rebuild
    print("\nSearch after automatic rebuild:")
    results, _ = splade_service.search(collection_id, query, top_k=10)
    print(f"  Found {len(results)} results")
    
    # Clean up settings
    settings.SOFT_DELETE_ENABLED = original_setting
    settings.INDEX_REBUILD_THRESHOLD = original_threshold
    

def main():
    """Main test function"""
    print("=" * 50)
    print("FAISS INDEX TYPES AND SOFT DELETION TEST")
    print("=" * 50)
    
    # Test different FAISS index types
    test_faiss_index_types()
    
    # Test soft deletion
    test_soft_deletion()
    
    print("\nTests completed!")


if __name__ == "__main__":
    main()
