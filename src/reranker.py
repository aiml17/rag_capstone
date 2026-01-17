"""
Cross-Encoder Reranker for RAG Pipeline
Reranks retrieved documents using a cross-encoder model for improved relevance.
"""

import os
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Disable telemetry
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

import torch
from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    """Single reranked result."""
    content: str
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval results.

    Cross-encoders process query-document pairs together, allowing for
    better relevance scoring than bi-encoders (used in dense retrieval).
    """

    # Available reranker models
    RERANKER_MODELS = {
        'ms-marco-mini': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'ms-marco-small': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'ms-marco-base': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',
        'bge-reranker': 'BAAI/bge-reranker-base',
        'bge-reranker-large': 'BAAI/bge-reranker-large',
    }

    def __init__(
            self,
            model_name: str = 'ms-marco-mini',
            device: Optional[str] = None,
            verbose: bool = False
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Name of reranker model (see RERANKER_MODELS)
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
            verbose: Print debug info
        """
        self.verbose = verbose

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        # Load model
        model_path = self.RERANKER_MODELS.get(model_name, model_name)

        if verbose:
            print(f"Loading reranker: {model_path} on {device}")

        self.model = CrossEncoder(model_path, device=device)

        if verbose:
            print(f"CrossEncoderReranker initialized: {model_name}")

    def rerank(
            self,
            query: str,
            documents: List[str],
            original_scores: Optional[List[float]] = None,
            top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of document strings to rerank
            original_scores: Optional original retrieval scores
            top_k: Number of results to return (None = all)

        Returns:
            List of RerankResult objects sorted by rerank score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Handle original scores
        if original_scores is None:
            original_scores = [0.0] * len(documents)

        # Create results with both scores
        results = []
        for i, (doc, rerank_score) in enumerate(zip(documents, scores)):
            results.append({
                'content': doc,
                'original_score': original_scores[i] if i < len(original_scores) else 0.0,
                'rerank_score': float(rerank_score),
                'original_rank': i + 1
            })

        # Sort by rerank score (descending)
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Apply top_k
        if top_k is not None:
            results = results[:top_k]

        # Convert to RerankResult objects
        output = []
        for new_rank, r in enumerate(results):
            output.append(RerankResult(
                content=r['content'],
                original_score=r['original_score'],
                rerank_score=r['rerank_score'],
                original_rank=r['original_rank'],
                new_rank=new_rank + 1
            ))

        return output

    def rerank_with_retrieval_results(
            self,
            query: str,
            retrieval_results: List,  # List of RetrievalResult or similar
            top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank results from a retriever.

        Args:
            query: Search query
            retrieval_results: Results from retriever (must have .content and .score)
            top_k: Number of results to return

        Returns:
            List of RerankResult objects
        """
        documents = [r.content for r in retrieval_results]
        scores = [r.score for r in retrieval_results]

        return self.rerank(query, documents, scores, top_k)


class HybridRetrieverWithReranker:
    """
    Complete retrieval pipeline: Hybrid Retrieval + Cross-Encoder Reranking
    """

    def __init__(
            self,
            dense_model: str = 'minilm',
            sparse_method: str = 'tfidf',
            reranker_model: str = 'ms-marco-mini',
            alpha: float = 0.5,
            initial_top_k: int = 20,  # Get more candidates for reranking
            verbose: bool = False
    ):
        """
        Initialize hybrid retriever with reranker.

        Args:
            dense_model: Dense embedding model name
            sparse_method: 'bm25' or 'tfidf'
            reranker_model: Cross-encoder model name
            alpha: Weight for dense retrieval
            initial_top_k: Number of candidates to retrieve before reranking
            verbose: Print debug info
        """
        from hybrid_retriever import HybridRetriever

        self.hybrid_retriever = HybridRetriever(
            dense_model=dense_model,
            sparse_method=sparse_method,
            alpha=alpha,
            verbose=verbose
        )

        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
            verbose=verbose
        )

        self.initial_top_k = initial_top_k
        self.verbose = verbose

        if verbose:
            print(f"HybridRetrieverWithReranker initialized")
            print(f"  Initial retrieval: top-{initial_top_k}")
            print(f"  Reranker: {reranker_model}")

    def index(self, documents: List[str]) -> None:
        """Index documents for retrieval."""
        self.hybrid_retriever.index(documents)

    def retrieve(
            self,
            query: str,
            top_k: int = 5,
            fusion_method: str = 'rrf'
    ) -> List[RerankResult]:
        """
        Retrieve and rerank documents.

        Args:
            query: Search query
            top_k: Final number of results to return
            fusion_method: 'rrf' or 'linear' for hybrid fusion

        Returns:
            List of RerankResult objects
        """
        # Step 1: Get initial candidates from hybrid retriever
        initial_results = self.hybrid_retriever.retrieve(
            query,
            top_k=self.initial_top_k,
            fusion_method=fusion_method
        )

        if self.verbose:
            print(f"  Retrieved {len(initial_results)} candidates")

        # Step 2: Rerank with cross-encoder
        reranked = self.reranker.rerank_with_retrieval_results(
            query,
            initial_results,
            top_k=top_k
        )

        if self.verbose:
            print(f"  Reranked to top-{len(reranked)}")

        return reranked


def test_reranker():
    """Test the cross-encoder reranker."""
    print("=" * 60)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 60)

    # Sample documents
    documents = [
        "Apple Inc. reported quarterly revenue of $89.5 billion.",
        "The company announced new iPhone models at the event.",
        "Apple's services segment grew 14% year over year.",
        "Tim Cook discussed the company's AI strategy.",
        "Revenue from Greater China declined 8% this quarter.",
    ]

    # Initialize reranker
    reranker = CrossEncoderReranker(
        model_name='ms-marco-mini',
        verbose=True
    )

    # Test query
    query = "What was Apple's revenue this quarter?"

    print(f"\nQuery: {query}")
    print("-" * 40)

    # Original order (simulating retrieval results)
    print("\nOriginal order:")
    for i, doc in enumerate(documents):
        print(f"  [{i + 1}] {doc[:60]}...")

    # Rerank
    results = reranker.rerank(query, documents)

    print("\nAfter reranking:")
    for r in results:
        print(f"  [{r.new_rank}] Score: {r.rerank_score:.4f} (was rank {r.original_rank})")
        print(f"      {r.content[:60]}...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_reranker()