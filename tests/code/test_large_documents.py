#!/usr/bin/env python3
"""
Test script for evaluating how the chunker handles extremely large documents.
This script creates a simulated large document (140 pages) with tables and tests the chunking.
"""

import os
import sys
import time

# Add parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.chunking import document_chunker
from app.models.schema import Document


def generate_large_document_with_tables(pages=140):
    """Generate a simulated large document with tables, approximately the size of 140 pages"""
    # Approximately 500 words per page = 70,000 words for 140 pages
    
    # Create document content
    content = "# Large Document Test\n\n"
    content += "This is a test document simulating a very large document with tables.\n\n"
    
    # Add sections with heading levels
    for section in range(1, 11):  # 10 major sections
        content += f"# Section {section}\n\n"
        content += f"This is the introduction to section {section}.\n\n"
        
        # Add subsections
        for subsection in range(1, 5):  # 4 subsections per section
            content += f"## Section {section}.{subsection}\n\n"
            content += f"This is subsection {section}.{subsection} content.\n\n"
            
            # Add paragraphs
            for para in range(1, 4):  # 3 paragraphs per subsection
                content += generate_paragraph(words=100, section=section, subsection=subsection, para=para)
                content += "\n\n"
            
            # Add a small table to some subsections
            if subsection % 2 == 0:
                content += generate_table(rows=5, cols=4, section=section, subsection=subsection)
                content += "\n\n"
        
        # Add a large table to some sections
        if section % 3 == 0:
            content += "### Large Table Example\n\n"
            content += generate_table(rows=50, cols=5, section=section, large=True)
            content += "\n\n"
        
        # Add a very large table to one section
        if section == 8:
            content += "### Very Large Table Example\n\n"
            content += generate_table(rows=150, cols=6, section=section, large=True)
            content += "\n\n"
    
    # Create a document
    doc = Document(
        id="large-doc-test",
        content=content,
        metadata={"format": "markdown", "type": "test", "pages": pages}
    )
    
    print(f"Generated a document with {len(content)} characters, approximately {pages} pages")
    
    return doc


def generate_paragraph(words=100, section=1, subsection=1, para=1):
    """Generate a paragraph with specified number of words"""
    base_words = [
        "document", "content", "large", "table", "section", "analysis", 
        "data", "information", "research", "study", "report", "results",
        "method", "approach", "process", "system", "test", "evaluation",
        "structure", "format", "technical", "specification", "requirement",
        "design", "implementation", "performance", "quality", "standard",
        "benchmark", "comparison", "assessment", "measurement", "metric"
    ]
    
    # Create paragraph text
    text = f"This is paragraph {para} in subsection {section}.{subsection}. "
    
    # Add words to reach the target
    for i in range(words - 10):  # Subtract approx words already added
        text += base_words[i % len(base_words)] + " "
        
        # Add some periods to create sentences
        if i % 15 == 14:
            text += ". "
    
    return text.strip()


def generate_table(rows=10, cols=4, section=1, subsection=1, large=False):
    """Generate a markdown table with specified dimensions"""
    # Create header row
    header = "| "
    for col in range(1, cols + 1):
        header += f"Column {col} | "
    
    # Create separator row
    separator = "| "
    for col in range(cols):
        separator += "-------- | "
    
    # Create data rows
    data_rows = []
    for row in range(1, rows + 1):
        data_row = "| "
        for col in range(1, cols + 1):
            # Make cell content larger for large tables to test token limits
            if large:
                data_row += f"Data {section}.{row}.{col} with additional text to increase size | "
            else:
                data_row += f"Data {section}.{row}.{col} | "
        data_rows.append(data_row)
    
    # Combine all parts
    table = header + "\n" + separator + "\n" + "\n".join(data_rows)
    
    return table


def measure_chunking_performance(document):
    """Measure performance of chunking and analyze results"""
    start_time = time.time()
    chunks = document_chunker.chunk_document(document)
    end_time = time.time()
    
    # Calculate stats
    chunk_count = len(chunks)
    avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / chunk_count if chunk_count > 0 else 0
    processing_time = end_time - start_time
    
    # Count tables in chunks
    table_info = []
    for i, chunk in enumerate(chunks):
        # Count pipe characters as rough indicator of table rows
        pipe_count = chunk.content.count("|")
        if pipe_count > 0:
            # Check if chunk contains a table separator row
            has_separator = bool(chunk.content.count("----"))
            table_rows = pipe_count // (has_separator + 1)  # Rough estimate
            table_info.append({
                "chunk_index": i,
                "table_rows": table_rows,
                "has_separator": has_separator,
                "chunk_size": len(chunk.content)
            })
    
    # Print results
    print("\nChunking Performance:")
    print(f"Total chunks: {chunk_count}")
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Chunks per second: {chunk_count / processing_time:.2f}")
    
    print("\nTable Statistics:")
    print(f"Total chunks with tables: {len(table_info)}")
    if table_info:
        print(f"Average table rows per chunk: {sum(t['table_rows'] for t in table_info) / len(table_info):.1f}")
        print(f"Largest table chunk: {max(t['chunk_size'] for t in table_info)} characters")
        
        # Find tables that were split
        section_chunks = {}
        for chunk in chunks:
            if "section_index" in chunk.metadata:
                section_idx = chunk.metadata["section_index"]
                if section_idx not in section_chunks:
                    section_chunks[section_idx] = []
                section_chunks[section_idx].append(chunk)
        
        # Look for sections with multiple table chunks
        sections_with_split_tables = []
        for section_idx, section_chunk_list in section_chunks.items():
            table_chunks = [i for i, c in enumerate(section_chunk_list) 
                          if c.content.count("|") > 0 and "large" in c.content.lower()]
            if len(table_chunks) > 1:
                sections_with_split_tables.append((section_idx, len(table_chunks)))
        
        if sections_with_split_tables:
            print("\nDetected split tables:")
            for section_idx, count in sections_with_split_tables:
                print(f"  Section {section_idx}: {count} table chunks")
    
    return chunks, {
        "chunk_count": chunk_count,
        "avg_chunk_size": avg_chunk_size,
        "processing_time": processing_time,
        "table_chunks": len(table_info)
    }


def examine_large_table_handling(chunks):
    """Examine how large tables were handled specifically"""
    # Find chunks with "Large Table Example" or "Very Large Table Example"
    large_table_chunks = []
    very_large_table_chunks = []
    
    for i, chunk in enumerate(chunks):
        if "Large Table Example" in chunk.content:
            if "Very Large Table Example" in chunk.content:
                very_large_table_chunks.append(i)
            else:
                large_table_chunks.append(i)
    
    # Print information about them
    print("\nLarge Table Handling:")
    print(f"Large table example chunks: {len(large_table_chunks)}")
    print(f"Very large table example chunks: {len(very_large_table_chunks)}")
    
    # Check for header repetition in split tables
    for chunk_type, chunk_indices in [
        ("Large", large_table_chunks), 
        ("Very Large", very_large_table_chunks)
    ]:
        if len(chunk_indices) > 1:
            print(f"\n{chunk_type} table was split across {len(chunk_indices)} chunks")
            
            # Check for header repetition in split tables
            header_count = 0
            for idx in chunk_indices:
                chunk_content = chunks[idx].content
                header_lines = [line for line in chunk_content.split("\n") if "Column" in line and "|" in line]
                header_count += len(header_lines)
            
            if header_count > 1:
                print(f"  Table headers were successfully repeated ({header_count} headers for {len(chunk_indices)} chunks)")
            else:
                print("  WARNING: Table headers were not properly repeated")
            
            # Check for table note in split tables
            note_count = 0
            for idx in chunk_indices:
                chunk_content = chunks[idx].content
                if "Table split due to size" in chunk_content:
                    note_count += 1
            
            if note_count > 0:
                print(f"  Split table notes were added to {note_count} of {len(chunk_indices)} chunks")


def main():
    """Main test function"""
    print("=" * 50)
    print("LARGE DOCUMENT CHUNKING TEST")
    print("=" * 50)
    
    # Generate a large test document
    document = generate_large_document_with_tables(pages=140)
    
    # Measure chunking performance
    chunks, stats = measure_chunking_performance(document)
    
    # Examine large table handling
    examine_large_table_handling(chunks)
    
    # Print sample chunks
    print("\nSample Chunks:")
    
    # Print a few chunks at different positions
    sample_indices = [0]  # First chunk
    if len(chunks) > 1:
        sample_indices.append(len(chunks) // 2)  # Middle chunk
    if len(chunks) > 2:
        sample_indices.append(len(chunks) - 1)  # Last chunk
    
    for idx in sample_indices:
        print(f"\nChunk {idx} (ID: {chunks[idx].id}):")
        print("-" * 40)
        content = chunks[idx].content
        print(content[:200] + "..." if len(content) > 200 else content)
        print("-" * 40)
    
    # Print conclusions
    print("\nTest Conclusions:")
    if stats["chunk_count"] > 0:
        print(f"✓ Successfully chunked a large document ({document.metadata['pages']} pages)")
        print(f"✓ Created {stats['chunk_count']} chunks in {stats['processing_time']:.2f} seconds")
        if stats["table_chunks"] > 0:
            print(f"✓ Preserved table structure in {stats['table_chunks']} chunks")
    else:
        print("✗ Failed to chunk the document properly")


if __name__ == "__main__":
    main()
