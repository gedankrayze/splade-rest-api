# Using Pagination with the SPLADE Content Server API

When working with large document collections in the SPLADE Content Server, it's important to efficiently retrieve and
navigate through search results. The API now supports pagination for all search endpoints, allowing you to:

- Limit the number of results returned in a single request
- Navigate through large result sets page by page
- Obtain metadata about the total number of results and pages

This article explains how to use pagination with the SPLADE Content Server API and provides examples of common use
cases.

## Pagination Parameters

The following pagination parameters are available for all search endpoints:

| Parameter   | Type    | Description                                   | Default                 |
|-------------|---------|-----------------------------------------------|-------------------------|
| `page`      | integer | The page number to retrieve (starting from 1) | 1                       |
| `page_size` | integer | Number of results per page                    | Same as `top_k` setting |

### Response Format

When using pagination, the API response includes a `pagination` object with the following information:

```json
{
  "results": [...],
  "query_time_ms": 23.45,
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_results": 45,
    "total_pages": 5
  }
}
```

The pagination object provides:

- `page`: Current page number
- `page_size`: Number of results per page
- `total_results`: Total number of results matching your query
- `total_pages`: Total number of pages available

## Example Usage

### Basic Pagination

To retrieve the second page of search results with 5 results per page:

```bash
curl -X GET "http://localhost:3000/search/technical-docs?query=neural%20networks&page=2&page_size=5"
```

### Combine with Other Parameters

Pagination works seamlessly with other search parameters:

```bash
# Paginated search with metadata filtering
curl -X GET "http://localhost:3000/search/technical-docs?query=machine%20learning&page=2&page_size=10&metadata_filter=%7B%22category%22%3A%22AI%22%7D"

# Paginated advanced search with deduplication
curl -X GET "http://localhost:3000/advanced-search/technical-docs?query=transformer%20models&page=3&page_size=10&min_score=0.5&deduplicate=true"
```

### Cross-Collection Search

When searching across all collections, each collection's results are paginated independently:

```bash
curl -X GET "http://localhost:3000/search?query=deep%20learning&page=2&page_size=5"
```

Response format for cross-collection searches:

```json
{
  "results": {
    "collection-1": [...],
    "collection-2": [...]
  },
  "pagination": {
    "collection-1": {
      "page": 2,
      "page_size": 5,
      "total_results": 25,
      "total_pages": 5
    },
    "collection-2": {
      "page": 2,
      "page_size": 5,
      "total_results": 15,
      "total_pages": 3
    }
  },
  "query_time_ms": 45.67
}
```

## Implementing Pagination in Your Client

Here's a JavaScript example of implementing pagination in a client application:

```javascript
async function searchWithPagination(query, page = 1, pageSize = 10) {
  const response = await fetch(
    `http://localhost:3000/search/technical-docs?query=${encodeURIComponent(query)}&page=${page}&page_size=${pageSize}`
  );
  
  const data = await response.json();
  
  // Display results
  displayResults(data.results);
  
  // Update pagination controls
  updatePaginationControls(data.pagination);
}

function updatePaginationControls(pagination) {
  const { page, page_size, total_results, total_pages } = pagination;
  
  // Update UI elements - example implementation
  document.getElementById('current-page').textContent = page;
  document.getElementById('total-pages').textContent = total_pages;
  document.getElementById('total-results').textContent = total_results;
  
  // Disable/enable previous/next buttons
  document.getElementById('prev-btn').disabled = page <= 1;
  document.getElementById('next-btn').disabled = page >= total_pages;
}

// Example event handlers
document.getElementById('prev-btn').addEventListener('click', () => {
  const currentPage = parseInt(document.getElementById('current-page').textContent);
  if (currentPage > 1) {
    searchWithPagination(currentQuery, currentPage - 1, currentPageSize);
  }
});

document.getElementById('next-btn').addEventListener('click', () => {
  const currentPage = parseInt(document.getElementById('current-page').textContent);
  const totalPages = parseInt(document.getElementById('total-pages').textContent);
  if (currentPage < totalPages) {
    searchWithPagination(currentQuery, currentPage + 1, currentPageSize);
  }
});
```

## Best Practices

1. **Choose an appropriate page size**: Smaller page sizes result in faster response times but require more API calls to
   navigate through all results.

2. **Cache previous results**: To improve user experience, consider caching previously loaded pages to allow quick
   navigation between pages.

3. **Handle empty results gracefully**: If a user navigates to a page beyond the available results or a search returns
   no matches, ensure your application displays an appropriate message.

4. **Adjust effective result count**: When using advanced search with deduplication or high minimum score thresholds,
   you might get fewer results than expected. Consider dynamically adjusting your pagination UI based on the actual
   number of results returned.

5. **Consider sorting options**: While the current implementation sorts by relevance, you might want to add other
   sorting options in your application logic when displaying paginated results.

## Performance Considerations

The pagination implementation in SPLADE Content Server is designed with performance in mind:

- The system retrieves more results than needed for the current page to properly calculate total counts
- For advanced search, the server applies filtering, deduplication, and thresholding before pagination
- When searching across collections, each collection's results are paginated independently to optimize memory usage

However, there are some things to keep in mind:

- Requesting very large page sizes can increase response time and memory usage
- When working with extremely large collections (millions of documents), consider using smaller page sizes
- The first page request will typically take longer than subsequent page requests since it calculates the total count

## Frequently Asked Questions

### How do I know if there are more results?

Check the `total_pages` value in the pagination response. If `current_page < total_pages`, there are more results
available.

### Why is my total result count different from what I expected?

When using advanced search with deduplication or high minimum score thresholds, the number of results may be lower than
the total number of documents that match a query. The pagination information reflects the final number of results after
all filtering and processing.

### Can I skip pagination and get all results?

For performance reasons, it's recommended to use pagination even if you want to retrieve all results. However, if you
need all results at once for a specific use case, you can set a very large `page_size` value, but be aware of potential
performance implications.

### How does pagination interact with chunking?

When using the advanced search endpoint with `merge_chunks=true`, chunks from the same document are merged before
pagination is applied. This means that pagination counts are based on the number of unique documents, not individual
chunks.

## Conclusion

The pagination feature in the SPLADE Content Server API makes it easier to work with large result sets efficiently. By
using these pagination parameters, your applications can provide a better user experience when navigating through search
results while maintaining good performance.