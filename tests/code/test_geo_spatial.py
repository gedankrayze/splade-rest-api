#!/usr/bin/env python3
"""
Test script for geo-spatial query functionality
This script creates test documents with location information and tests various geo-spatial queries.
"""

import os
import sys
import time
import random
import math

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.splade_service import splade_service
from app.models.schema import Document, GeoCoordinates
from app.core.geo_spatial_index import GeoSpatialIndex


def create_test_points_in_area(center_lat, center_lon, radius_km, count):
    """Create random test points within a radius around a center point"""
    points = []
    
    # Earth's radius in km
    earth_radius = 6371.0
    
    # Convert radius from km to radians
    radius_rad = radius_km / earth_radius
    
    # Convert center coordinates to radians
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)
    
    for i in range(count):
        # Generate random point in circle
        # Random distance within radius
        distance = radius_rad * math.sqrt(random.random())
        # Random angle
        angle = 2 * math.pi * random.random()
        
        # Calculate new position
        lat_rad = math.asin(math.sin(center_lat_rad) * math.cos(distance) + 
                           math.cos(center_lat_rad) * math.sin(distance) * math.cos(angle))
        lon_rad = center_lon_rad + math.atan2(math.sin(angle) * math.sin(distance) * math.cos(center_lat_rad),
                                             math.cos(distance) - math.sin(center_lat_rad) * math.sin(lat_rad))
        
        # Convert back to degrees
        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)
        
        # Ensure valid coordinates
        lat = max(min(lat, 90.0), -90.0)
        lon = (lon + 540) % 360 - 180  # Normalize to -180 to 180
        
        # Calculate actual distance to verify
        actual_distance = GeoSpatialIndex()._haversine_distance(center_lat, center_lon, lat, lon)
        
        points.append({
            "lat": lat,
            "lon": lon,
            "distance": actual_distance
        })
    
    return points


def test_geo_spatial_search():
    """Test geo-spatial search functionality"""
    # Create a test collection
    collection_id = "geo-test-collection"
    
    # Create or clear collection
    if collection_id in [c["id"] for c in splade_service.list_collections()]:
        # Delete existing collection
        splade_service.delete_collection(collection_id)
        
    splade_service.create_collection(
        collection_id=collection_id,
        name="Geo-Spatial Test Collection",
        description="Collection for testing geo-spatial search"
    )
    
    # Define a center point (San Francisco)
    sf_lat, sf_lon = 37.7749, -122.4194
    
    # Create test points in different areas
    # Points within 5km of San Francisco
    sf_points = create_test_points_in_area(sf_lat, sf_lon, 5, 20)
    
    # Points within 5km of New York
    ny_lat, ny_lon = 40.7128, -74.0060
    ny_points = create_test_points_in_area(ny_lat, ny_lon, 5, 20)
    
    # Points within 5km of Tokyo
    tokyo_lat, tokyo_lon = 35.6762, 139.6503
    tokyo_points = create_test_points_in_area(tokyo_lat, tokyo_lon, 5, 20)
    
    # Create and add documents
    all_documents = []
    
    print("Creating test documents...")
    
    # San Francisco area documents
    for i, point in enumerate(sf_points):
        doc = Document(
            id=f"sf-poi-{i}",
            content=f"This is point of interest #{i} in San Francisco. It's located {point['distance']:.2f}km from downtown.",
            metadata={"city": "San Francisco", "category": "poi"},
            location=GeoCoordinates(
                latitude=point["lat"],
                longitude=point["lon"]
            )
        )
        all_documents.append(doc)
    
    # New York area documents
    for i, point in enumerate(ny_points):
        doc = Document(
            id=f"ny-poi-{i}",
            content=f"This is point of interest #{i} in New York City. It's located {point['distance']:.2f}km from downtown.",
            metadata={"city": "New York", "category": "poi"},
            location=GeoCoordinates(
                latitude=point["lat"],
                longitude=point["lon"]
            )
        )
        all_documents.append(doc)
    
    # Tokyo area documents
    for i, point in enumerate(tokyo_points):
        doc = Document(
            id=f"tokyo-poi-{i}",
            content=f"This is point of interest #{i} in Tokyo. It's located {point['distance']:.2f}km from downtown.",
            metadata={"city": "Tokyo", "category": "poi"},
            location=GeoCoordinates(
                latitude=point["lat"],
                longitude=point["lon"]
            )
        )
        all_documents.append(doc)
    
    # Add all documents in batch
    print(f"Adding {len(all_documents)} documents with location data...")
    count = splade_service.batch_add_documents(collection_id, all_documents, auto_chunk=False)
    print(f"Successfully added {count} documents")
    
    # Run geo-spatial queries
    print("\nRunning geo-spatial queries...\n")
    
    # Test cases
    test_cases = [
        {
            "name": "San Francisco - 2km radius",
            "lat": sf_lat,
            "lon": sf_lon,
            "radius": 2,
            "query": "point of interest"
        },
        {
            "name": "San Francisco - 10km radius",
            "lat": sf_lat,
            "lon": sf_lon,
            "radius": 10,
            "query": "point of interest"
        },
        {
            "name": "New York - 3km radius",
            "lat": ny_lat,
            "lon": ny_lon,
            "radius": 3,
            "query": "point of interest"
        },
        {
            "name": "Tokyo - 5km radius + metadata",
            "lat": tokyo_lat,
            "lon": tokyo_lon,
            "radius": 5,
            "query": "point of interest",
            "metadata": {"city": "Tokyo"}
        },
        {
            "name": "Invalid location (middle of ocean)",
            "lat": 0,
            "lon": 0,
            "radius": 10,
            "query": "point of interest"
        }
    ]
    
    # Run each test case
    for test in test_cases:
        print(f"Test: {test['name']}")
        print(f"Query: '{test['query']}' within {test['radius']}km of ({test['lat']}, {test['lon']})")
        
        metadata_filter = test.get("metadata")
        
        # Create geo filter
        geo_filter = {
            "latitude": test["lat"],
            "longitude": test["lon"],
            "radius_km": test["radius"]
        }
        
        # Execute search
        start_time = time.time()
        results, query_time = splade_service.search(
            collection_id, 
            test["query"],
            top_k=10,
            filter_metadata=metadata_filter,
            geo_filter=geo_filter
        )
        total_time = (time.time() - start_time) * 1000
        
        # Print results
        print(f"Found {len(results)} results in {query_time:.2f}ms (total time: {total_time:.2f}ms)")
        
        if results:
            # Display sample results
            for i, result in enumerate(results[:3]):  # Show top 3
                print(f"  Result {i+1}: {result['id']} - Distance: {result.get('distance_km', 'N/A')}km, Score: {result['score']:.4f}")
                if result.get('location'):
                    print(f"    Location: {result['location']['latitude']}, {result['location']['longitude']}")
            
            # Verify distances
            distances = [result.get('distance_km', float('inf')) for result in results]
            max_distance = max(distances) if distances else 0
            avg_distance = sum(distances) / len(distances) if distances else 0
            
            print(f"  Max distance: {max_distance:.2f}km, Avg distance: {avg_distance:.2f}km")
            
            # Check if all results are within the requested radius
            all_in_radius = all(d <= test["radius"] * 1.01 for d in distances)  # Allow 1% margin for floating point errors
            print(f"  All results within radius: {'✓' if all_in_radius else '✗'}")
        
        print()
    
    # Test combined metadata and geo filtering
    print("\nTesting combined filtering (metadata + geo)...")
    
    # Search for Tokyo points with specific metadata
    results, query_time = splade_service.search(
        collection_id, 
        "point of interest",
        top_k=10,
        filter_metadata={"city": "Tokyo", "category": "poi"},
        geo_filter={
            "latitude": tokyo_lat,
            "longitude": tokyo_lon,
            "radius_km": 3
        }
    )
    
    print("Tokyo POIs within 3km with metadata filters:")
    print(f"Found {len(results)} results in {query_time:.2f}ms")
    
    if results:
        for i, result in enumerate(results[:3]):
            print(f"  Result {i+1}: {result['id']} - Distance: {result.get('distance_km', 'N/A')}km")
    
    # Return created collection for further testing
    return collection_id


def test_rest_api_geo_search():
    """Test geo-spatial search via REST API (simulation)"""
    
    # We'll simulate the API calls by calling the functions directly
    # This would normally use the requests library to call the actual API
    
    print("\nSimulating REST API geo-spatial search...\n")
    
    # San Francisco coordinates
    sf_lat, sf_lon = 37.7749, -122.4194
    
    # Example API Query URLs:
    example_urls = [
        f"/search/geo-test-collection?query=point+of+interest&latitude={sf_lat}&longitude={sf_lon}&radius_km=5",
        f"/search?query=poi&latitude={sf_lat}&longitude={sf_lon}&radius_km=10&metadata_filter=%7B%22category%22%3A%22poi%22%7D",
        f"/advanced-search/geo-test-collection?query=interest&latitude={sf_lat}&longitude={sf_lon}&radius_km=3&min_score=0.3"
    ]
    
    print("Example API URLs for geo-spatial search:")
    for url in example_urls:
        print(f"  GET {url}")
    
    print("\nNote: These URLs would be called via HTTP requests in a real application.")


def main():
    """Main test function"""
    print("=" * 50)
    print("GEO-SPATIAL SEARCH TEST")
    print("=" * 50)
    
    # Run tests
    test_geo_spatial_search()
    
    # Test REST API simulation
    test_rest_api_geo_search()
    
    print("\nAll tests completed!")
    

if __name__ == "__main__":
    main()
