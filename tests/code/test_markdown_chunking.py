#!/usr/bin/env python3
"""
Test script for the Markdown-aware document chunker.
This script tests how the chunker handles various Markdown structures.
"""

import os
import sys

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.chunking import document_chunker
from app.models.schema import Document


def print_chunks(title, chunks):
    """Print chunks with a title banner"""
    print("\n" + "=" * 40)
    print(f" {title} ")
    print("=" * 40)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (ID: {chunk.id}):")
        print("-" * 30)
        print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
        print("-" * 30)
        print(f"Estimated tokens: {document_chunker._estimate_tokens(chunk.content)}")


def test_markdown_table():
    """Test chunking with Markdown tables"""
    content = """
# Product Comparison

This document compares different products in our catalog.

## Table of Features

| Feature          | Product A | Product B | Product C |
|------------------|-----------|-----------|-----------|
| Price            | $100      | $200      | $150      |
| Weight           | 2.5 kg    | 3.0 kg    | 1.8 kg    |
| Battery Life     | 10 hours  | 15 hours  | 8 hours   |
| Waterproof       | Yes       | No        | Yes       |
| Warranty         | 1 year    | 2 years   | 1 year    |

Please note that all prices are subject to change without notice.

## Additional Information

Product A is our best-selling model, while Product B offers premium features.
Product C is the most portable option in our lineup.

| Performance Test | Product A | Product B | Product C |
|------------------|-----------|-----------|-----------|
| Speed            | Fast      | Very Fast | Medium    |
| Accuracy         | High      | Medium    | Very High |
| Durability       | Good      | Excellent | Good      |
| User Rating      | 4.5/5     | 4.2/5     | 4.7/5     |

Contact customer support for more details about any product.
"""
    
    document = Document(
        id="test-tables",
        content=content,
        metadata={"type": "markdown", "test": "tables"}
    )
    
    chunks = document_chunker.chunk_document(document)
    print_chunks("MARKDOWN TABLE TEST", chunks)
    
    # Verify table integrity in chunks
    for chunk in chunks:
        table_rows = chunk.content.count("|")
        if table_rows > 0:
            lines = chunk.content.split("\n")
            table_lines = [line for line in lines if line.strip().startswith("|")]
            
            # Check if headers and separator lines are together with data rows
            if any("|---" in line for line in table_lines):
                header_indices = [i for i, line in enumerate(table_lines) if "|---" in line]
                for idx in header_indices:
                    if idx == 0 or idx == len(table_lines) - 1:
                        print(f"[WARNING] Table separator line at boundary: {idx}")
    
    return chunks


def test_code_blocks():
    """Test chunking with Markdown code blocks"""
    content = """
# Code Examples

Here are some code examples in different languages.

## Python Example

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Calculate the 10th Fibonacci number
result = fibonacci(10)
print(f"The 10th Fibonacci number is: {result}")
```

This function calculates Fibonacci numbers efficiently.

## JavaScript Example

```javascript
function quickSort(arr) {
    if (arr.length <= 1) {
        return arr;
    }
    
    const pivot = arr[0];
    const left = [];
    const right = [];
    
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] < pivot) {
            left.push(arr[i]);
        } else {
            right.push(arr[i]);
        }
    }
    
    return [...quickSort(left), pivot, ...quickSort(right)];
}

// Example usage
const sortedArray = quickSort([3, 1, 4, 1, 5, 9, 2, 6, 5]);
console.log(sortedArray);
```

The quickSort algorithm is a divide-and-conquer sorting algorithm.

## SQL Example

```sql
SELECT 
    products.name,
    categories.name AS category,
    products.price,
    products.stock
FROM 
    products
JOIN 
    categories ON products.category_id = categories.id
WHERE 
    products.price > 100
    AND products.stock > 0
ORDER BY 
    products.price DESC
LIMIT 10;
```

This query retrieves the top 10 most expensive products that are in stock.
"""
    
    document = Document(
        id="test-code-blocks",
        content=content,
        metadata={"type": "markdown", "test": "code-blocks"}
    )
    
    chunks = document_chunker.chunk_document(document)
    print_chunks("MARKDOWN CODE BLOCKS TEST", chunks)
    
    # Verify code block integrity
    for chunk in chunks:
        code_start_count = chunk.content.count("```")
        if code_start_count % 2 != 0:
            print(f"[WARNING] Code block may be split in chunk: {chunk.id}")
    
    return chunks


def test_mixed_content():
    """Test chunking with mixed Markdown content"""
    content = """
# Mixed Markdown Content

This document contains a variety of Markdown elements to test chunking.

## Lists and Tables

Here's a list of items:

* Item 1
* Item 2
* Item 3
  * Subitem 3.1
  * Subitem 3.2
* Item 4

And here's a table:

| Name     | Age | Occupation   |
|----------|-----|--------------|
| John Doe | 32  | Developer    |
| Jane Doe | 28  | Designer     |
| Bob Smith| 45  | Manager      |

## Code and Blockquotes

A simple code example:

```python
def greet(name):
    return f"Hello, {name}!"
```

And a blockquote:

> This is a famous quote.
> It spans multiple lines.
> - Famous Person

## Formatting and Links

We can use **bold text**, *italic text*, or ***both***.

Here's a [link to Google](https://www.google.com) and another [link to GitHub](https://github.com).

### Images and Horizontal Rules

An image:

![Sample Image](https://example.com/sample.jpg)

---

This is below a horizontal rule.
"""
    
    document = Document(
        id="test-mixed",
        content=content,
        metadata={"type": "markdown", "test": "mixed"}
    )
    
    chunks = document_chunker.chunk_document(document)
    print_chunks("MIXED MARKDOWN TEST", chunks)
    
    return chunks


def test_long_paragraph():
    """Test chunking with a very long paragraph"""
    # Generate a long paragraph
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", 
             "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", 
             "et", "dolore", "magna", "aliqua"]
    
    long_para = " ".join([words[i % len(words)] for i in range(1000)])
    
    content = f"""
# Document with Long Paragraph

## Introduction

This is a short introduction.

## Main Content

{long_para}

## Conclusion

This is a short conclusion.
"""
    
    document = Document(
        id="test-long-paragraph",
        content=content,
        metadata={"type": "markdown", "test": "long-paragraph"}
    )
    
    chunks = document_chunker.chunk_document(document)
    print_chunks("LONG PARAGRAPH TEST", chunks)
    
    return chunks


def test_very_large_table():
    """Test chunking with a very large table"""
    # Generate a large table
    header = "| " + " | ".join([f"Column {i}" for i in range(1, 6)]) + " |"
    separator = "| " + " | ".join(["------" for i in range(5)]) + " |"
    
    rows = []
    for i in range(1, 101):  # 100 rows
        row = "| " + " | ".join([f"Value {i},{j}" for j in range(1, 6)]) + " |"
        rows.append(row)
    
    table = "\n".join([header, separator] + rows)
    
    content = f"""
# Document with Very Large Table

## Introduction

This document contains a very large table that should test the chunker's ability to handle large structured content.

## The Data

{table}

## Conclusion

This is the end of the document.
"""
    
    document = Document(
        id="test-large-table",
        content=content,
        metadata={"type": "markdown", "test": "large-table"}
    )
    
    chunks = document_chunker.chunk_document(document)
    print_chunks("VERY LARGE TABLE TEST", chunks)
    
    # This is a special case - we expect the table to be split
    # because it's too large to fit in one chunk
    print(f"\nNumber of chunks: {len(chunks)}")
    table_row_counts = [chunk.content.count("|") // 2 for chunk in chunks]  # Approx rows
    print(f"Approx table rows per chunk: {table_row_counts}")
    
    return chunks


def main():
    """Run all tests"""
    test_markdown_table()
    test_code_blocks()
    test_mixed_content()
    test_long_paragraph()
    test_very_large_table()


if __name__ == "__main__":
    main()
