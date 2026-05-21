# LLM-Enhanced Search with SPLADE Content Server

The SPLADE Content Server now features powerful LLM-enhanced search capabilities, allowing you to get better search results and even direct answers to your questions based on your document collections. This article explains how to use these new features and how they can improve your search experience.

## New Features Overview

The SPLADE Content Server has been enhanced with two major LLM-powered features:

1. **Query Expansion**: Automatically expand your search queries to include synonyms, related terms, and alternative phrasings
2. **Question Answering**: Ask natural language questions and get comprehensive answers synthesized from your documents

These capabilities leverage large language models (LLMs) like GPT-4o-mini to provide more intelligent and helpful interactions with your document collections.

## Using Query Expansion

Query expansion helps overcome the vocabulary mismatch problem, where documents might contain relevant information but use different terminology than your search query. When enabled, the system uses an LLM to expand your original query with synonyms and related terms, improving search recall.

### How to Use Query Expansion

Simply add the `query_expansion=true` parameter to your existing `/advanced-search` endpoint:

```
GET /advanced-search/my-collection?query=heat pump dimensions&query_expansion=true
```

You can also specify a custom LLM model:

```
GET /advanced-search/my-collection?query=heat pump dimensions&query_expansion=true&expansion_model=gpt-4
```

### Response with Query Expansion

When query expansion is used, the response includes both the original and expanded query:

```json
{
  "results": [...],
  "query_time_ms": 187.5,
  "pagination": {...},
  "expanded_query": "heat pump dimensions measurements size specifications width height depth physical dimensions",
  "original_query": "heat pump dimensions"
}
```

## Using the Question Answering Endpoint

The new `/query` endpoint allows you to ask natural language questions and receive direct answers synthesized from your documents. This is particularly useful for extracting specific information across multiple document chunks.

### How to Use Question Answering

To ask a question about documents in a specific collection:

```
GET /query/my-collection?question=What are the dimensions of the CHA-07 heat pump?
```

You can also search across all collections:

```
GET /query?question=What are the dimensions of the CHA-07 heat pump?
```

Additional parameters include:

- `model`: Custom LLM model to use (default: gpt-4o-mini)
- `top_k`: Number of search results to consider
- `min_score`: Minimum similarity score threshold
- `query_expansion`: Whether to use query expansion (default: true)

### Response Format

The question answering endpoint returns both the synthesized answer and the search results:

```json
{
  "question": "What are the dimensions of the CHA-07 heat pump?",
  "answer": "The dimensions of the CHA-07 heat pump are not explicitly stated in the provided documentation. The document contains technical specifications for the CHA-07/400V heat pump, including performance data, temperatures, and energy efficiency ratings, but does not list the physical dimensions (width, height, depth) of the unit.",
  "search_results": [...],
  "query_time_ms": 215.3,
  "expanded_query": "dimensions size specifications measurements width height depth physical dimensions CHA-07 heat pump HVAC system"
}
```

## Benefits of LLM-Enhanced Search

1. **Improved Search Quality**: Automatically expand queries to find more relevant documents
2. **Natural Language Understanding**: Ask questions in plain language instead of crafting perfect keywords
3. **Information Synthesis**: Get comprehensive answers that combine information from multiple document chunks
4. **Time Saving**: Quickly extract the information you need without reading through all search results

## Technical Details

### LLM Integration Configuration

The LLM integration can be configured in the server settings:

```python
# In app/core/config.py
LLM_DEFAULT_MODEL: str = "gpt-4o-mini"  
LLM_ENABLED: bool = True
```

### Supported LLM Models

The system uses an OpenAI-compatible API interface, so it works with:

- OpenAI models (gpt-4o, gpt-3.5-turbo, etc.)
- Ollama models (llama3, mistral, etc.)
- Any other OpenAI API-compatible service

### Performance Considerations

LLM-enhanced features add some latency to search operations:

- Query expansion adds approximately 1-2 seconds to search time
- Question answering adds approximately 2-4 seconds to search time

The system is designed to handle errors gracefully, falling back to standard search when LLM services are unavailable.

## Example Use Cases

### Technical Documentation Search

When searching technical documentation like product manuals, query expansion helps find relevant information even when terminology varies:

```
Query: "CHA-07 startup procedure"
Expanded: "CHA-07 startup procedure initialization commissioning first start setup initial operation power on sequence"
```

### Customer Support Knowledge Base

When answering customer questions, the query endpoint can synthesize information across multiple support articles:

```
Question: "How do I reset my heat pump after a power outage?"
Answer: "After a power outage, you can reset your heat pump by following these steps: 1) Check that power has been restored to the unit, 2) Wait at least 3 minutes to allow internal pressures to stabilize, 3) Use the control panel to reset any error codes by pressing the 'Reset' button, 4) If error codes persist, consult the troubleshooting section of your manual or contact support."
```

## Conclusion

The LLM-enhanced search features provide a significant upgrade to the SPLADE Content Server's capabilities, making it easier to find relevant information and extract answers from your document collections. They build on the hybrid retrieval pipeline: SPLADE finds candidates, FastEmbed dense vectors rerank them, and the LLM layer can expand queries or synthesize answers from the retrieved context.
