"""
Enhanced document chunking service for processing Markdown and other structured documents,
with special handling for large documents and tables
"""

import logging
import re
from typing import List, Dict, Any, Tuple

from app.models.schema import Document

# Configure logging
logger = logging.getLogger(__name__)


class MarkdownAwareChunker:
    """Service for chunking documents with awareness of Markdown structures like tables"""

    def __init__(
            self,
            max_chunk_size: int = 500,  # Tokens (approximate)
            table_chunk_size: int = 1000,  # Larger limit for tables
            overlap_size: int = 50,  # Tokens of overlap between chunks
            avg_word_length: int = 5,  # Average word length to estimate tokens
            max_section_level: int = 2  # Maximum header level to consider as major section
    ):
        """Initialize markdown-aware document chunker with options for large documents"""
        self.max_chunk_size = max_chunk_size
        self.table_chunk_size = table_chunk_size
        self.overlap_size = overlap_size
        self.avg_word_length = avg_word_length
        self.max_section_level = max_section_level

        # Patterns for identifying special Markdown elements
        self.table_pattern = re.compile(r'(\|[^\n]+\|\n)((?:\|[-:| ]+\|\n))((?:\|[^\n]+\|\n)+)')
        self.code_block_pattern = re.compile(r'```(?:.*?)\n(.*?)```', re.DOTALL)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.*?)$', re.MULTILINE)

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks based on semantic boundaries
        with special handling for Markdown structures
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks with appropriate metadata
        """
        content = document.content

        # If content is small enough, return as is
        if self._estimate_tokens(content) <= self.max_chunk_size:
            return [document]

        # Initialize metadata for chunks
        base_metadata = document.metadata or {}

        # For very large documents, first segment into major sections
        if self._estimate_tokens(content) > self.max_chunk_size * 10:
            logger.info(f"Large document detected (ID: {document.id}). Segmenting into major sections first.")
            sections = self._segment_into_sections(content)

            # Process each section independently
            all_chunks = []
            section_index = 0

            for section_title, section_content in sections:
                # Extend metadata with section information
                section_metadata = base_metadata.copy()
                if section_title:
                    section_metadata.update({
                        "section_title": section_title,
                        "section_index": section_index
                    })

                # Process blocks within this section
                blocks = self._extract_semantic_blocks(section_content)
                section_chunks = self._create_chunks_from_blocks(
                    document, blocks, section_index * 1000, section_metadata
                )
                all_chunks.extend(section_chunks)
                section_index += 1

            return all_chunks
        else:
            # For smaller documents, process directly with block extraction
            blocks = self._extract_semantic_blocks(content)
            return self._create_chunks_from_blocks(document, blocks, 0, base_metadata)

    def _segment_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Segment a large document into major sections based on headers
        Returns a list of (section_title, section_content) tuples
        """
        # Find all headers in the document
        headers = list(self.header_pattern.finditer(content))

        # If no headers or very few, treat as a single section
        if len(headers) <= 1:
            return [("", content)]

        # Identify major section headers (up to max_section_level)
        sections = []

        for i, header_match in enumerate(headers):
            level = len(header_match.group(1))  # Number of # characters
            title = header_match.group(2).strip()

            # Only consider headers up to max_section_level as section boundaries
            if level <= self.max_section_level:
                start_pos = header_match.start()

                # Find the end of this section (next major header or end of document)
                end_pos = len(content)
                for j in range(i + 1, len(headers)):
                    next_level = len(headers[j].group(1))
                    if next_level <= self.max_section_level:
                        end_pos = headers[j].start()
                        break

                # Extract section content
                section_content = content[start_pos:end_pos]
                sections.append((title, section_content))

        # If no major sections found, treat as a single section
        if not sections:
            return [("", content)]

        return sections

    def _create_chunks_from_blocks(
            self,
            document: Document,
            blocks: List[str],
            start_index: int,
            metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Create document chunks from a list of content blocks
        """
        chunks = []
        current_chunk = ""
        chunk_index = start_index

        for block in blocks:
            # Check if this is a table block that needs special handling
            is_table = bool(self.table_pattern.search(block))

            # Use appropriate chunk size limit based on content type
            effective_max_size = self.table_chunk_size if is_table else self.max_chunk_size

            block_tokens = self._estimate_tokens(block.strip())

            # If block itself is too large, split it if possible
            if block_tokens > effective_max_size:
                # Add the current chunk if it's not empty
                if current_chunk.strip():
                    chunk_doc = self._create_chunk_document(
                        document, current_chunk, chunk_index, metadata
                    )
                    chunks.append(chunk_doc)
                    chunk_index += 1

                # Handle splitting based on content type
                if is_table:
                    table_chunks = self._split_large_table(block)
                    for table_chunk in table_chunks:
                        if not table_chunk.strip():
                            continue
                        chunk_doc = self._create_chunk_document(
                            document, table_chunk, chunk_index, metadata
                        )
                        chunks.append(chunk_doc)
                        chunk_index += 1
                else:
                    # Split other large blocks (paragraphs, etc.)
                    sub_blocks = self._split_large_block(block)
                    for sub_block in sub_blocks:
                        if not sub_block.strip():
                            continue
                        chunk_doc = self._create_chunk_document(
                            document, sub_block, chunk_index, metadata
                        )
                        chunks.append(chunk_doc)
                        chunk_index += 1

                current_chunk = ""
                continue

            chunk_tokens = self._estimate_tokens(current_chunk)

            # If adding block exceeds chunk size and we already have content,
            # finish current chunk and start a new one
            if chunk_tokens + block_tokens > self.max_chunk_size and chunk_tokens > 0:
                # Create document for current chunk
                chunk_doc = self._create_chunk_document(
                    document, current_chunk, chunk_index, metadata
                )
                chunks.append(chunk_doc)

                # Start new chunk with overlap
                if current_chunk and not self._is_special_structure(block):
                    # Get approximate overlap text from the end of the previous chunk
                    words = current_chunk.split()
                    overlap_word_count = min(self.overlap_size // 2, len(words))
                    overlap_text = " ".join(words[-overlap_word_count:]) if overlap_word_count > 0 else ""
                    current_chunk = overlap_text + "\n\n" + block if overlap_text else block
                else:
                    current_chunk = block

                chunk_index += 1
            else:
                # Add block to current chunk
                current_chunk = current_chunk + "\n\n" + block if current_chunk else block

        # Add the last chunk if there's any content left
        if current_chunk.strip():
            chunk_doc = self._create_chunk_document(
                document, current_chunk, chunk_index, metadata
            )
            chunks.append(chunk_doc)

        return chunks

    def _extract_semantic_blocks(self, content: str) -> List[str]:
        """
        Extract semantically meaningful blocks from content,
        preserving special structures like tables and code blocks
        """
        # Special block markers
        SPECIAL_MARKER = "_SPECIAL_BLOCK_"
        special_blocks = []

        # Find and store tables
        content_with_markers = content

        # Extract tables
        table_matches = list(self.table_pattern.finditer(content))
        for i, match in enumerate(reversed(table_matches)):  # Process in reverse to not disrupt indices
            marker = f"{SPECIAL_MARKER}{len(special_blocks)}"
            special_blocks.append(match.group(0))
            start, end = match.span()
            content_with_markers = content_with_markers[:start] + marker + content_with_markers[end:]

        # Extract code blocks
        code_matches = list(self.code_block_pattern.finditer(content_with_markers))
        for i, match in enumerate(reversed(code_matches)):  # Process in reverse to not disrupt indices
            marker = f"{SPECIAL_MARKER}{len(special_blocks)}"
            special_blocks.append(match.group(0))
            start, end = match.span()
            content_with_markers = content_with_markers[:start] + marker + content_with_markers[end:]

        # Split by paragraphs
        blocks = re.split(r'\n\s*\n', content_with_markers)

        # Replace markers with original content
        result_blocks = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Replace special markers
            for i, special_block in enumerate(special_blocks):
                marker = f"{SPECIAL_MARKER}{i}"
                if marker in block:
                    block = block.replace(marker, special_block)

            result_blocks.append(block)

        return result_blocks

    def _is_special_structure(self, text: str) -> bool:
        """Check if text contains a special structure like table or code block"""
        return bool(self.table_pattern.search(text) or self.code_block_pattern.search(text))

    def _split_large_table(self, table_content: str) -> List[str]:
        """
        Split a large table into smaller chunks while preserving structure
        Repeats header rows in each chunk for context
        """
        # Extract header and separator rows
        lines = table_content.strip().split('\n')

        # Ensure we have at least header, separator, and one data row
        if len(lines) < 3:
            return [table_content]  # Table is too small to split

        header = lines[0] if lines else ""
        separator = lines[1] if len(lines) > 1 else ""

        # Add metadata about table splitting
        table_note = "<!-- Table split due to size -->\n"

        # Group rows into chunks, repeating header for each chunk
        chunks = []
        current_chunk = [header, separator]
        current_size = self._estimate_tokens(header + '\n' + separator)

        for line in lines[2:]:  # Skip header and separator
            line_size = self._estimate_tokens(line)

            if current_size + line_size > self.table_chunk_size:
                if len(current_chunk) > 2:  # Only if we have content beyond header
                    chunk_content = '\n'.join(current_chunk)
                    if chunks:  # Not the first chunk
                        chunk_content = table_note + chunk_content
                    chunks.append(chunk_content)
                # Start new chunk with header and separator
                current_chunk = [header, separator, line]
                current_size = self._estimate_tokens(header + '\n' + separator + '\n' + line)
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add the last chunk if there's content
        if len(current_chunk) > 2:  # Only if we have content beyond header
            chunk_content = '\n'.join(current_chunk)
            if chunks:  # Not the first chunk
                chunk_content = table_note + chunk_content
            chunks.append(chunk_content)

        return chunks

    def _split_large_block(self, block: str) -> List[str]:
        """Split a large text block into smaller pieces, respecting sentence boundaries"""
        # Use sentence splitting for large blocks
        sentences = re.split(r'(?<=[.!?])\s+', block)

        sub_blocks = []
        current_sub_block = ""

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            current_tokens = self._estimate_tokens(current_sub_block)

            # If this single sentence is too large, split by comma or just add it
            if sentence_tokens > self.max_chunk_size:
                # If we have content, add it as a sub-block
                if current_sub_block:
                    sub_blocks.append(current_sub_block)
                    current_sub_block = ""

                # Try to split by comma
                comma_parts = re.split(r'(?<=,)\s+', sentence)

                # If even comma parts are too big, just add the whole sentence
                if any(self._estimate_tokens(part) > self.max_chunk_size for part in comma_parts):
                    sub_blocks.extend(self._split_large_text(sentence))
                else:
                    current_part = ""
                    for part in comma_parts:
                        part_tokens = self._estimate_tokens(part)
                        current_part_tokens = self._estimate_tokens(current_part)

                        if current_part_tokens + part_tokens <= self.max_chunk_size:
                            current_part = current_part + ", " + part if current_part else part
                        else:
                            sub_blocks.append(current_part)
                            current_part = part

                    if current_part:
                        sub_blocks.append(current_part)

            # Normal case - add to current sub-block or start a new one
            elif current_tokens + sentence_tokens <= self.max_chunk_size:
                current_sub_block = current_sub_block + " " + sentence if current_sub_block else sentence
            else:
                sub_blocks.append(current_sub_block)
                current_sub_block = sentence

        # Add the last sub-block
        if current_sub_block:
            sub_blocks.append(current_sub_block)

        return sub_blocks

    def _split_large_text(self, text: str) -> List[str]:
        """Split text with no useful semantic boundaries into bounded word chunks."""
        words = text.split()
        if not words:
            return []

        max_words = max(1, int(self.max_chunk_size / 1.3))
        overlap_words = min(max(0, self.overlap_size // 2), max_words - 1)
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + max_words, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap_words

        return chunks

    def _create_chunk_document(
            self,
            parent: Document,
            content: str,
            chunk_index: int,
            base_metadata: Dict[str, Any]
    ) -> Document:
        """Create a document chunk with appropriate metadata"""
        # Copy and extend metadata
        metadata = base_metadata.copy()
        metadata.update({
            "original_document_id": parent.id,
            "chunk_index": chunk_index,
            "is_chunk": True
        })

        # Generate a deterministic ID for the chunk
        chunk_id = f"{parent.id}_chunk_{chunk_index}"

        return Document(
            id=chunk_id,
            content=content,
            metadata=metadata
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text based on words and characters"""
        if not text:
            return 0

        # Basic estimation: count words and add character-based adjustment
        words = len(text.split())

        # A rough estimation: each word is about 1.3 tokens on average
        estimated_tokens = words * 1.3

        return int(estimated_tokens)
