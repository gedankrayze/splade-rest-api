#!/usr/bin/env python3
"""
Example script demonstrating domain-specific model usage
"""

import sys

from memsplora_client import MemSploraClient


def main():
    # Initialize the client
    client = MemSploraClient("http://localhost:3000")

    try:
        # Create collections with different models

        # Standard collection using default model
        standard_collection = create_test_collection(
            client,
            "standard-collection",
            "Standard Collection",
            "Using default SPLADE model",
            model_name=None  # Use default model
        )

        # Domain-specific collection
        domain_collection = create_test_collection(
            client,
            "domain-collection",
            "Domain Collection",
            "Using domain-specific SPLADE model",
            model_name="domain-splade"  # Specify domain model
        )

        # Add test documents to both collections
        documents = [
            {
                "id": "doc1",
                "content": "This is a test document about artificial intelligence and machine learning techniques.",
                "metadata": {"category": "AI"}
            },
            {
                "id": "doc2",
                "content": "Domain-specific language models can improve search relevance for specialized content.",
                "metadata": {"category": "NLP"}
            }
        ]

        # Add documents to collections
        for collection_id in [standard_collection["id"], domain_collection["id"]]:
            print(f"\nAdding documents to {collection_id}:")

            for i, doc in enumerate(documents):
                # Create a unique ID for each doc in each collection
                doc_copy = doc.copy()
                doc_copy["id"] = f"{collection_id}-{doc['id']}"

                try:
                    result = client.add_document(collection_id, doc_copy)
                    print(f"  Added document {doc_copy['id']}")
                except Exception as e:
                    print(f"  Error adding document: {e}")

        # Perform searches in each collection
        query = "domain specific models"

        for collection_id in [standard_collection["id"], domain_collection["id"]]:
            print(f"\nSearching in {collection_id} for '{query}':")

            try:
                # The query will be encoded with the appropriate model for each collection
                results = client.search(collection_id, query, top_k=5)

                print(f"  Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")

                for i, result in enumerate(results["results"]):
                    print(f"  Result {i + 1}: {result['id']} - Score: {result['score']:.4f}")
                    print(f"    {result['content'][:100]}...")
            except Exception as e:
                print(f"  Error searching: {e}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def create_test_collection(client, collection_id, name, description, model_name=None):
    """Create a test collection, deleting any existing one with the same ID"""

    # Check if collection exists, delete if it does
    try:
        client.get_collection(collection_id)
        print(f"Deleting existing collection {collection_id}")
        client.delete_collection(collection_id)
    except Exception:
        pass  # Collection doesn't exist

    # Create new collection
    print(f"Creating collection {collection_id}" +
          (f" with model {model_name}" if model_name else " with default model"))

    collection = client.create_collection(
        collection_id=collection_id,
        name=name,
        description=description,
        model_name=model_name
    )

    print(f"Collection created: {collection['name']}")
    return collection


if __name__ == "__main__":
    main()
