"""
Document chunking service for processing large documents
"""

import re
from typing import List, Dict, Any

from app.core.markdown_chunker import MarkdownAwareChunker
from app.models.schema import Document


class DocumentChunker:
    """Service for chunking large documents into smaller pieces for better processing"""

    def __init__(
            self,
            max_chunk_size: int = 500,  # Tokens (approximate)
            overlap_size: int = 50,  # Tokens of overlap between chunks
            avg_word_length: int = 5  # Average word length to estimate tokens
    ):
        """Initialize document chunker"""
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.avg_word_length = avg_word_length

        # Initialize markdown-aware chunker
        self.markdown_chunker = MarkdownAwareChunker(
            max_chunk_size=max_chunk_size,
            overlap_size=overlap_size,
            avg_word_length=avg_word_length
        )

        # Regular expressions for content type detection
        self.markdown_patterns = [
            re.compile(r'^\s*#'),  # Headings
            re.compile(r'\|[-:| ]+\|'),  # Tables
            re.compile(r'```'),  # Code blocks
            re.compile(r'\*\*.*?\*\*'),  # Bold
            re.compile(r'\*.*?\*'),  # Italic
            re.compile(r'\[.*?\]\(.*?\)'),  # Links
            re.compile(r'^>\s'),  # Blockquotes
            re.compile(r'^\s*[-*+]\s'),  # Lists
            re.compile(r'^\s*\d+\.\s')  # Numbered lists
        ]

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks based on semantic boundaries
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks with appropriate metadata
        """
        # Check if the content is Markdown
        is_markdown = self._detect_markdown(document.content)

        # Use appropriate chunking strategy
        if is_markdown:
            return self.markdown_chunker.chunk_document(document)
        else:
            return self._chunk_plain_text(document)

    def _detect_markdown(self, content: str) -> bool:
        """
        Detect if the content is Markdown by looking for common Markdown patterns
        """
        if not content:
            return False

        # Check for common Markdown patterns
        for pattern in self.markdown_patterns:
            if pattern.search(content):
                return True

        return False

    def _chunk_plain_text(self, document: Document) -> List[Document]:
        """
        Original chunking method for plain text content
        """
        content = document.content

        # If content is small enough, return as is
        if self._estimate_tokens(content) <= self.max_chunk_size:
            return [document]

        # Initialize metadata for chunks
        base_metadata = document.metadata or {}

        # Split content by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', content)

        # Create chunks by combining paragraphs up to max size
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue

            para_tokens = self._estimate_tokens(para)
            chunk_tokens = self._estimate_tokens(current_chunk)

            if para_tokens > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(self._create_chunk_document(document, current_chunk, chunk_index, base_metadata))
                    chunk_index += 1
                    current_chunk = ""

                for part in self._split_large_text(para):
                    chunks.append(self._create_chunk_document(document, part, chunk_index, base_metadata))
                    chunk_index += 1
                continue

            # If adding paragraph exceeds chunk size and we already have content,
            # finish current chunk and start a new one
            if chunk_tokens + para_tokens > self.max_chunk_size and chunk_tokens > 0:
                # Create document for current chunk
                chunk_doc = self._create_chunk_document(
                    document, current_chunk, chunk_index, base_metadata
                )
                chunks.append(chunk_doc)

                # Start new chunk with overlap
                if current_chunk:
                    # Get approximate overlap text from the end of the previous chunk
                    words = current_chunk.split()
                    overlap_word_count = min(self.overlap_size // 2, len(words))
                    overlap_text = " ".join(words[-overlap_word_count:]) if overlap_word_count > 0 else ""
                    current_chunk = overlap_text + " " + para if overlap_text else para
                else:
                    current_chunk = para

                chunk_index += 1
            else:
                # Add paragraph to current chunk
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        # Add the last chunk if there's any content left
        if current_chunk.strip():
            chunk_doc = self._create_chunk_document(
                document, current_chunk, chunk_index, base_metadata
            )
            chunks.append(chunk_doc)

        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """Split text that has no paragraph boundaries into bounded word chunks."""
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


# Create a singleton instance
document_chunker = DocumentChunker()
