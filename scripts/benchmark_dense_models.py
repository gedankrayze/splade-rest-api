#!/usr/bin/env python3
"""
Benchmark FastEmbed dense models against the sample docs/ corpus.

The labels are weak but practical: each query has one or more expected sample
documents. The goal is to compare dense model quality and latency before
choosing a default reranker.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding


DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

LARGE_MODELS = [
    "intfloat/multilingual-e5-large",
]

DEFAULT_QUERIES = [
    ("geo spatial radius search by coordinates", {"geo_spatial_search.md"}),
    ("find nearby restaurants using latitude longitude distance", {"geo_spatial_search.md"}),
    ("split large markdown tables without breaking headers", {"large_document_handling.md", "chunking_and_deduplication.md"}),
    ("deduplicate chunks from the same source document", {"chunking_and_deduplication.md"}),
    ("handle 140 page documents and preserve table structure", {"large_document_handling.md"}),
    ("choose FAISS flat IVF HNSW index type", {"performance_optimizations.md"}),
    ("soft deletion delayed index rebuild threshold", {"performance_optimizations.md"}),
    ("use different SPLADE models for legal medical collections", {"domain_specific_models.md"}),
    ("metadata filters with semantic search", {"geo_spatial_search.md", "chunking_and_deduplication.md"}),
    ("optimize search latency and indexing performance", {"performance_optimizations.md"}),
]


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Query:
    text: str
    expected_doc_ids: set[str]


@dataclass
class BenchmarkResult:
    model: str
    doc_count: int
    query_count: int
    embed_docs_ms: float
    p50_query_ms: float
    p95_query_ms: float
    recall_at_1: float
    recall_at_3: float
    mrr_at_10: float
    ndcg_at_10: float


def load_documents(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        documents.append(Document(doc_id=path.name, text=path.read_text(encoding="utf-8")))
    if not documents:
        raise ValueError(f"No markdown documents found in {docs_dir}")
    return documents


def normalize(vectors: Iterable[np.ndarray]) -> list[np.ndarray]:
    normalized = []
    for vector in vectors:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        normalized.append(array if norm == 0 else array / norm)
    return normalized


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def ranked_doc_ids(query_vector: np.ndarray, documents: list[Document], doc_vectors: list[np.ndarray]) -> list[str]:
    scores = np.asarray([float(np.dot(query_vector, doc_vector)) for doc_vector in doc_vectors])
    ranked_indices = np.argsort(-scores)
    return [documents[index].doc_id for index in ranked_indices]


def recall_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    return 1.0 if expected.intersection(ranked[:k]) else 0.0


def reciprocal_rank_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    score = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in expected:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    ideal_hits = min(len(expected), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(ranked, expected, k) / ideal_dcg


def benchmark_model(model_name: str, documents: list[Document], queries: list[Query]) -> BenchmarkResult:
    model = TextEmbedding(model_name=model_name)

    start = time.perf_counter()
    doc_vectors = normalize(model.embed([document.text for document in documents]))
    embed_docs_ms = (time.perf_counter() - start) * 1000

    query_latencies = []
    recall_1 = []
    recall_3 = []
    mrr_10 = []
    ndcg_10 = []

    for query in queries:
        start = time.perf_counter()
        query_vector = normalize(model.embed([query.text]))[0]
        ranked = ranked_doc_ids(query_vector, documents, doc_vectors)
        query_latencies.append((time.perf_counter() - start) * 1000)

        recall_1.append(recall_at_k(ranked, query.expected_doc_ids, 1))
        recall_3.append(recall_at_k(ranked, query.expected_doc_ids, 3))
        mrr_10.append(reciprocal_rank_at_k(ranked, query.expected_doc_ids, 10))
        ndcg_10.append(ndcg_at_k(ranked, query.expected_doc_ids, 10))

    return BenchmarkResult(
        model=model_name,
        doc_count=len(documents),
        query_count=len(queries),
        embed_docs_ms=embed_docs_ms,
        p50_query_ms=statistics.median(query_latencies),
        p95_query_ms=percentile(query_latencies, 95),
        recall_at_1=statistics.mean(recall_1),
        recall_at_3=statistics.mean(recall_3),
        mrr_at_10=statistics.mean(mrr_10),
        ndcg_at_10=statistics.mean(ndcg_10),
    )


def print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "model",
        "docs",
        "queries",
        "embed_docs_ms",
        "p50_ms",
        "p95_ms",
        "recall@1",
        "recall@3",
        "mrr@10",
        "ndcg@10",
    ]
    rows = [
        [
            result.model,
            str(result.doc_count),
            str(result.query_count),
            f"{result.embed_docs_ms:.1f}",
            f"{result.p50_query_ms:.1f}",
            f"{result.p95_query_ms:.1f}",
            f"{result.recall_at_1:.3f}",
            f"{result.recall_at_3:.3f}",
            f"{result.mrr_at_10:.3f}",
            f"{result.ndcg_at_10:.3f}",
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(BenchmarkResult.__annotations__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FastEmbed dense models against docs/ sample data.")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"), help="Directory with markdown sample docs.")
    parser.add_argument("--model", action="append", dest="models", help="FastEmbed model name. Repeatable.")
    parser.add_argument("--include-large", action="store_true", help="Also benchmark larger multilingual models.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.docs_dir)
    queries = [Query(text=text, expected_doc_ids=expected) for text, expected in DEFAULT_QUERIES]

    models = args.models or DEFAULT_MODELS.copy()
    if args.include_large:
        models.extend(LARGE_MODELS)

    results = []
    for model_name in models:
        print(f"Benchmarking {model_name}...")
        results.append(benchmark_model(model_name, documents, queries))

    results.sort(key=lambda item: (item.ndcg_at_10, item.recall_at_3, -item.p95_query_ms), reverse=True)
    print()
    print_table(results)

    if args.csv:
        write_csv(args.csv, results)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
