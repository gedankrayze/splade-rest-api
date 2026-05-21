from app.core.chunking import document_chunker
from app.models.schema import Document


def test_small_document_is_not_chunked():
    document = Document(
        id="small-doc",
        content="# Title\n\nShort content.",
        metadata={"type": "markdown"},
    )

    chunks = document_chunker.chunk_document(document)

    assert chunks == [document]


def test_large_paragraph_is_split_into_chunks():
    document = Document(
        id="large-doc",
        content=" ".join(f"word-{index}" for index in range(1_500)),
        metadata={"type": "plain"},
    )

    chunks = document_chunker.chunk_document(document)

    assert len(chunks) > 1
    assert all(chunk.metadata["original_document_id"] == "large-doc" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_metadata_latitude_longitude_populates_location():
    document = Document(
        id="geo-doc",
        content="Geo document",
        metadata={"latitude": 52.52, "longitude": 13.405},
    )

    assert document.location is not None
    assert document.location.to_tuple() == (52.52, 13.405)
