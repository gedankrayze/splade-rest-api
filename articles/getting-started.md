# Getting Started with SPLADE Content Server

SPLADE Content Server is a powerful solution for semantic search with efficient sparse representations. This article
will guide you through setting up and using the SPLADE Content Server for your search applications.

## What is SPLADE?

SPLADE (SParse Lexical AnD Expansion) is a neural information retrieval model that combines the efficiency of sparse
retrieval with semantic term expansion. This server uses SPLADE as the first retrieval stage, then reranks candidates
with local FastEmbed dense embeddings. SPLADE creates sparse vector representations that:

- Are highly interpretable
- Work efficiently with inverted indexes
- Perform lexical matching and semantic expansion
- Scale well to large document collections

## Key Features of SPLADE Content Server

- **Collection-based organization**: Group related documents into separate collections
- **In-memory operation**: Fast retrieval with memory-optimized indexes
- **FAISS integration**: Efficient similarity search using Facebook AI Similarity Search
- **Automatic persistence**: Changes are automatically saved to disk
- **Deduplication and chunking**: Process large documents intelligently
- **Geo-spatial search**: Find documents based on geographic location
- **Domain-specific models**: Use different models for different collections
- **Hybrid reranking**: Rerank SPLADE candidates with local FastEmbed dense vectors
- **RESTful API**: Easy integration with any application or service
- **Pagination**: Navigate through large result sets efficiently

## Installation

### Prerequisites

- Python 3.10 to 3.12
- `uv`
- 4GB+ of RAM (8GB+ recommended for larger collections)
- GPU support is optional but recommended for faster encoding

### Setup Steps

1. Clone the repository:

```bash
git clone https://github.com/gedankrayze/splade_rest_api.git
cd splade_rest_api
```

2. Install dependencies:

```bash
uv sync
```

3. Run the server:

```bash
task start
```

The server will start on `http://localhost:3000`. The first time you index or search, it automatically downloads the
default SPLADE model and FastEmbed dense model unless configured otherwise.

## Core Concepts

### Collections

Collections are containers for related documents. They allow you to:

- Organize content by topic, source, or purpose
- Apply different settings to different document sets
- Search within specific content domains

### Documents

Documents are the basic unit of content. Each document:

- Has a unique ID within its collection
- Contains the text content to be searched
- Can include metadata for filtering
- Can optionally have geographical coordinates

### Search

SPLADE Content Server provides several search capabilities:

- **Basic search**: Find relevant documents using semantic understanding
- **Advanced search**: Add deduplication, chunking, and threshold controls
- **Hybrid reranking**: Use SPLADE for candidate generation and FastEmbed dense embeddings for reranking
- **Cross-collection search**: Search across multiple collections at once
- **Geo-spatial search**: Find documents near specific coordinates
- **Metadata filtering**: Narrow results based on document attributes

## Basic Usage Examples

### Creating a Collection

```bash
curl -X POST "http://localhost:3000/collections" \
     -H "Content-Type: application/json" \
     -d '{"id": "articles", "name": "News Articles", "description": "Collection of news articles from various sources"}'
```

### Adding Documents

```bash
curl -X POST "http://localhost:3000/documents/articles" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "article-1",
       "content": "Scientists discover new renewable energy source that could revolutionize power generation.",
       "metadata": {
         "category": "Science",
         "author": "Jane Smith",
         "published_date": "2023-04-15"
       }
     }'
```

### Batch Adding Documents

```bash
curl -X POST "http://localhost:3000/documents/articles/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": [
         {
           "id": "article-2",
           "content": "Global economic outlook shows signs of recovery despite ongoing challenges.",
           "metadata": {"category": "Business", "author": "John Doe"}
         },
         {
           "id": "article-3",
           "content": "New diplomatic agreements aim to reduce tensions in the region.",
           "metadata": {"category": "Politics", "author": "Sarah Johnson"}
         }
       ]
     }'
```

### Basic Search

```bash
curl -X GET "http://localhost:3000/search/articles?query=renewable%20energy&top_k=5"
```

### Search with Metadata Filtering

```bash
curl -X GET "http://localhost:3000/search/articles?query=economic&metadata_filter=%7B%22category%22%3A%22Business%22%7D"
```

### Search with Pagination

```bash
curl -X GET "http://localhost:3000/search/articles?query=diplomacy&page=1&page_size=10"
```

## Advanced Configuration

SPLADE Content Server can be configured through environment variables or a `.env` file:

```
# Core Settings
SPLADE_MODEL_NAME=splade-cocondenser-selfdistil
SPLADE_MODEL_HF_ID=baseplate/splade-cocondenser-selfdistil
SPLADE_DATA_DIR=app/data
SPLADE_DEFAULT_TOP_K=10

# Performance Settings
SPLADE_FAISS_INDEX_TYPE=flat  # Options: flat, ivf, hnsw
SPLADE_SOFT_DELETE_ENABLED=true
SPLADE_INDEX_REBUILD_THRESHOLD=100
SPLADE_DENSE_RERANK_ENABLED=true
SPLADE_DENSE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SPLADE_HYBRID_MAX_CANDIDATES=200

# Chunking Settings
SPLADE_MAX_CHUNK_SIZE=500
SPLADE_CHUNK_OVERLAP=50
```

## Using Different FAISS Index Types

FAISS provides different index types for different use cases:

- **Flat**: Exact search, best for small to medium collections (default)
- **IVF**: Inverted file structure for faster approximate search with larger collections
- **HNSW**: Hierarchical navigable small world graph for very large collections

To change the index type:

```bash
# Set environment variable
export SPLADE_FAISS_INDEX_TYPE=hnsw

# Or in .env file
SPLADE_FAISS_INDEX_TYPE=hnsw
```

## Custom Domain-Specific Models

For specialized domains (legal, medical, technical, etc.), you can use domain-specific SPLADE models:

1. Train or fine-tune a domain-specific model using Hugging Face
2. Place it in the `models/{model_name}` directory
3. Create a collection with the specific model:

```bash
curl -X POST "http://localhost:3000/collections" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "legal-docs", 
       "name": "Legal Documents", 
       "description": "Legal contracts and documents", 
       "model_name": "legal-splade"
     }'
```

## Benchmarking Dense Rerankers

The server intentionally skips BM25 and uses SPLADE plus FastEmbed dense reranking. To compare dense models against the
sample `docs/` corpus:

```bash
task benchmark-dense
```

The default benchmark compares multilingual MiniLM against multilingual MPNet. MiniLM remains the default because it is
fast and multilingual. To include heavier models such as multilingual E5 large:

```bash
uv run python scripts/benchmark_dense_models.py --include-large
```

## Integration Examples

### Python Client

```python
import requests
import json

# Base URL of the SPLADE Content Server
base_url = "http://localhost:3000"


# Create a collection
def create_collection(collection_id, name, description=None):
    response = requests.post(
        f"{base_url}/collections",
        json={"id": collection_id, "name": name, "description": description}
    )
    return response.json()


# Add a document
def add_document(collection_id, doc_id, content, metadata=None):
    response = requests.post(
        f"{base_url}/documents/{collection_id}",
        json={"id": doc_id, "content": content, "metadata": metadata or {}}
    )
    return response.status_code == 201


# Search for documents
def search(collection_id, query, top_k=10, page=1, page_size=10, metadata_filter=None):
    params = {
        "query": query,
        "top_k": top_k,
        "page": page,
        "page_size": page_size
    }

    if metadata_filter:
        params["metadata_filter"] = json.dumps(metadata_filter)

    response = requests.get(
        f"{base_url}/search/{collection_id}",
        params=params
    )
    return response.json()


# Example usage
if __name__ == "__main__":
    # Create a collection
    create_collection("sample", "Sample Collection", "A sample collection for testing")

    # Add some documents
    add_document("sample", "doc1", "SPLADE is a sparse lexical model for information retrieval",
                 {"category": "AI", "type": "Model"})
    add_document("sample", "doc2", "Information retrieval systems help find relevant documents",
                 {"category": "IR", "type": "Concept"})

    # Search with pagination
    results = search("sample", "information retrieval", page=1, page_size=5,
                     metadata_filter={"category": "IR"})

    print(f"Found {results['pagination']['total_results']} results")
    for item in results["results"]:
        print(f"Document: {item['id']}")
        print(f"Score: {item['score']}")
        print(f"Content: {item['content'][:100]}...")
        print("---")
```

### JavaScript Client

```javascript
// SPLADE Content Server API Client
class SpladeClient {
  constructor(baseUrl = 'http://localhost:3000') {
    this.baseUrl = baseUrl;
  }

  // Collections
  async createCollection(id, name, description = '') {
    const response = await fetch(`${this.baseUrl}/collections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, name, description })
    });
    return response.json();
  }

  async listCollections() {
    const response = await fetch(`${this.baseUrl}/collections`);
    return response.json();
  }

  // Documents
  async addDocument(collectionId, document) {
    const response = await fetch(`${this.baseUrl}/documents/${collectionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(document)
    });
    return response.json();
  }

  async batchAddDocuments(collectionId, documents) {
    const response = await fetch(`${this.baseUrl}/documents/${collectionId}/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ documents })
    });
    return response.json();
  }

  // Search
  async search(collectionId, query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...options
    });
    
    const response = await fetch(`${this.baseUrl}/search/${collectionId}?${params}`);
    return response.json();
  }

  async advancedSearch(collectionId, query, options = {}) {
    const params = new URLSearchParams({
      query,
      ...options
    });
    
    const response = await fetch(`${this.baseUrl}/advanced-search/${collectionId}?${params}`);
    return response.json();
  }
}

// Example usage
const client = new SpladeClient();

async function demo() {
  // Create a collection
  await client.createCollection('blog-posts', 'Blog Posts', 'Collection of blog posts');
  
  // Add documents
  await client.addDocument('blog-posts', {
    id: 'post-1',
    content: 'How to implement semantic search in modern applications',
    metadata: { tags: ['search', 'tutorial'], author: 'Alice' }
  });
  
  // Search with pagination
  const searchResults = await client.search('blog-posts', 'semantic search', {
    page: 1,
    page_size: 10,
    metadata_filter: JSON.stringify({ tags: 'search' })
  });
  
  console.log('Search results:', searchResults);
}

demo();
```

## Next Steps

Now that you've got the basics of SPLADE Content Server, here are some next steps to explore:

1. **Experiment with different collections**: Create collections for different content types or domains
2. **Try advanced search options**: Explore chunking, deduplication, and score thresholding
3. **Implement geo-spatial search**: Add location data to your documents for proximity-based queries
4. **Benchmark dense rerankers**: Compare FastEmbed models against your own query set
5. **Customize for your domain**: Use or train domain-specific SPLADE models for better results
6. **Optimize performance**: Experiment with different FAISS index types and hybrid candidate counts

For more examples to index and search, use the tracked sample documents in the `docs/` directory of the repository.

Happy searching with SPLADE Content Server!
