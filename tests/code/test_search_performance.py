#!/usr/bin/env python3
"""
Test script for evaluating search performance with large documents.
This script measures search accuracy and performance across different document sizes.
"""

import os
import sys
import time
import json

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.splade_service import splade_service
from tests.test_large_documents import generate_large_document_with_tables


def create_test_collection():
    """Create a test collection with various document sizes"""
    collection_id = "search-performance-test"
    
    # Create collection if it doesn't exist
    try:
        splade_service.get_collection(collection_id)
        print(f"Using existing collection: {collection_id}")
    except Exception:
        print(f"Creating new collection: {collection_id}")
        splade_service.create_collection(
            collection_id=collection_id,
            name="Search Performance Test",
            description="Collection for testing search performance with large documents"
        )
    
    # Generate documents of different sizes
    documents = [
        generate_large_document_with_tables(pages=5),
        generate_large_document_with_tables(pages=20),
        generate_large_document_with_tables(pages=50),
        generate_large_document_with_tables(pages=140)
    ]
    
    # Update document IDs to indicate size
    for doc in documents:
        pages = doc.metadata["pages"]
        doc.id = f"test-doc-{pages}pages"
    
    # Add documents to collection
    for doc in documents:
        try:
            print(f"Adding document: {doc.id} ({len(doc.content)} chars)")
            result = splade_service.add_document(collection_id, doc, auto_chunk=True)
            if result:
                print(f"Successfully added: {doc.id}")
            else:
                print(f"Failed to add: {doc.id}")
        except Exception as e:
            print(f"Error adding document {doc.id}: {str(e)}")
    
    return collection_id


def perform_search_tests(collection_id):
    """Perform various search tests on the collection"""
    # Test queries
    test_queries = [
        {
            "query": "large table example",
            "description": "Finding large tables",
            "expected_in_results": ["test-doc-140pages", "test-doc-50pages"]
        },
        {
            "query": "section 8 subsection 2",
            "description": "Finding specific section with tables",
            "expected_in_results": ["test-doc-140pages", "test-doc-50pages", "test-doc-20pages"]
        },
        {
            "query": "very large table with column 5",
            "description": "Finding content in large tables",
            "expected_in_results": ["test-doc-140pages"]
        },
        {
            "query": "paragraph in subsection information data",
            "description": "General content search",
            "expected_in_results": ["test-doc-140pages", "test-doc-50pages", "test-doc-20pages", "test-doc-5pages"]
        }
    ]
    
    results = []
    
    # Run searches with different parameters
    search_configs = [
        {"mode": "basic", "params": {}},
        {"mode": "advanced", "params": {"deduplicate": True, "merge_chunks": False}},
        {"mode": "advanced", "params": {"deduplicate": True, "merge_chunks": True}}
    ]
    
    for config in search_configs:
        print(f"\nRunning {config['mode']} search tests:")
        
        for test in test_queries:
            query = test["query"]
            expected = test["expected_in_results"]
            
            print(f"\nSearch: '{query}'")
            start_time = time.time()
            
            if config["mode"] == "basic":
                search_results, query_time = splade_service.search(
                    collection_id, query, top_k=10
                )
            else:
                # Use advanced search API directly
                from app.core.search_utils import deduplicate_and_threshold_results, merge_chunk_content
                
                # Get raw results
                raw_results, query_time = splade_service.search(
                    collection_id, query, top_k=30  # Get more for processing
                )
                
                # Apply post-processing
                processed = deduplicate_and_threshold_results(
                    raw_results,
                    min_score_threshold=0.3,
                    deduplicate_by_original_id=config["params"]["deduplicate"]
                )
                
                # Merge chunks if requested
                if config["params"]["merge_chunks"]:
                    search_results = merge_chunk_content(processed)[:10]  # Limit to 10
                else:
                    search_results = processed[:10]  # Limit to 10
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Check if expected documents are in results
            found_docs = set()
            result_info = []
            
            for i, result in enumerate(search_results):
                result_id = result["id"]
                original_id = result.get("metadata", {}).get("original_document_id", result_id)
                
                # Check if this is from an expected document
                for exp_id in expected:
                    if exp_id in original_id or exp_id == original_id:
                        found_docs.add(exp_id)
                
                # Get snippet for result
                content = result["content"]
                snippet = content[:100] + "..." if len(content) > 100 else content
                
                result_info.append({
                    "position": i + 1,
                    "id": result_id,
                    "original_id": original_id if original_id != result_id else None,
                    "score": result["score"],
                    "snippet": snippet
                })
            
            # Calculate accuracy
            accuracy = len(found_docs) / len(expected) if expected else 1.0
            
            test_result = {
                "query": query,
                "description": test["description"],
                "mode": config["mode"],
                "params": config["params"],
                "expected_docs": expected,
                "found_docs": list(found_docs),
                "accuracy": accuracy,
                "api_time_ms": query_time,
                "total_time_ms": total_time * 1000,
                "result_count": len(search_results),
                "top_results": result_info[:3]  # Include top 3 results
            }
            
            results.append(test_result)
            
            # Print summary
            print(f"  Found {len(found_docs)}/{len(expected)} expected docs (accuracy: {accuracy*100:.1f}%)")
            print(f"  API time: {query_time:.2f}ms, Total time: {total_time*1000:.2f}ms")
            print(f"  Top result: {result_info[0]['id'] if result_info else 'None'}")
    
    return results


def main():
    """Main test function"""
    print("=" * 50)
    print("SEARCH PERFORMANCE TEST WITH LARGE DOCUMENTS")
    print("=" * 50)
    
    # Create a test collection
    collection_id = create_test_collection()
    
    # Perform search tests
    test_results = perform_search_tests(collection_id)
    
    # Analyze results
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    # Group by search mode
    modes = {}
    for result in test_results:
        mode = result["mode"]
        if mode not in modes:
            modes[mode] = []
        modes[mode].append(result)
    
    # Print summary by mode
    for mode, results in modes.items():
        params = results[0]["params"]
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "default"
        
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
        avg_api_time = sum(r["api_time_ms"] for r in results) / len(results)
        avg_total_time = sum(r["total_time_ms"] for r in results) / len(results)
        
        print(f"\n{mode.title()} Search ({param_str}):")
        print(f"  Average accuracy: {avg_accuracy*100:.1f}%")
        print(f"  Average API time: {avg_api_time:.2f}ms")
        print(f"  Average total time: {avg_total_time:.2f}ms")
    
    # Calculate overall best mode
    best_mode = max(modes.items(), key=lambda x: sum(r["accuracy"] for r in x[1]) / len(x[1]))
    best_mode_name = best_mode[0]
    best_mode_params = best_mode[1][0]["params"]
    param_str = ", ".join(f"{k}={v}" for k, v in best_mode_params.items()) if best_mode_params else "default"
    
    print(f"\nBest overall mode: {best_mode_name.title()} Search ({param_str})")
    
    # Save results to file
    results_file = "search_performance_results.json"
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
