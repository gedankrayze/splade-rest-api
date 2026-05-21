import numpy as np

from app.core.splade_service import SpladeCollection
from app.core.splade_service import SpladeService
from app.models.schema import Document, GeoCoordinates


def test_batch_add_indexes_document_locations(tmp_path):
    collection = SpladeCollection(
        collection_id="geo-batch",
        name="Geo Batch",
        vocab_size=4,
        data_dir=str(tmp_path),
    )
    document = Document(
        id="berlin",
        content="Berlin office",
        location=GeoCoordinates(latitude=52.52, longitude=13.405),
    )

    added = collection.batch_add_documents([document], [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)])

    assert added == 1
    assert collection.geo_index.search_radius(52.52, 13.405, 1) == {"berlin"}


def test_batch_add_persists_dense_vectors(tmp_path):
    collection = SpladeCollection(
        collection_id="dense-batch",
        name="Dense Batch",
        vocab_size=4,
        data_dir=str(tmp_path),
    )
    document = Document(id="dense-doc", content="Dense document")
    dense_vector = np.array([0.0, 1.0], dtype=np.float32)

    added = collection.batch_add_documents(
        [document],
        [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)],
        [dense_vector],
    )

    assert added == 1
    np.testing.assert_array_equal(collection.dense_vectors["dense-doc"], dense_vector)


def test_dense_rerank_uses_cosine_scores(tmp_path):
    collection = SpladeCollection(
        collection_id="rerank",
        name="Rerank",
        vocab_size=4,
        data_dir=str(tmp_path),
    )
    collection.dense_vectors = {
        "sparse-first": np.array([1.0, 0.0], dtype=np.float32),
        "dense-first": np.array([0.0, 1.0], dtype=np.float32),
    }
    results = [
        {"id": "sparse-first", "content": "Sparse first", "metadata": {}, "score": 0.9},
        {"id": "dense-first", "content": "Dense first", "metadata": {}, "score": 0.1},
    ]
    service = SpladeService.__new__(SpladeService)
    service.encode_dense_texts = lambda texts: [np.array([0.0, 1.0], dtype=np.float32)]

    reranked = service._dense_rerank(collection, "query", results, top_k=2)

    assert [result["id"] for result in reranked] == ["dense-first", "sparse-first"]
    assert reranked[0]["dense_score"] == 1.0
    assert reranked[0]["sparse_score"] == 0.1
