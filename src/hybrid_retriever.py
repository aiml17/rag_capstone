"""
Hybrid Retriever: Combines Dense and Sparse Retrieval
Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.

Save this file as: src/hybrid_retriever.py
"""

import os
import sys
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    content: str
    score: float
    rank: int
    source: str  # 'dense', 'sparse', or 'hybrid'


class HybridRetriever:
    """
    Hybrid retriever combining dense (embedding) and sparse (BM25/TF-IDF) retrieval.
    Uses Reciprocal Rank Fusion (RRF) to combine rankings.
    """

    # Embedding model mapping
    EMBEDDING_MODELS = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'bge-base': 'BAAI/bge-base-en-v1.5',
        'bge-large': 'BAAI/bge-large-en-v1.5',
        'e5-base': 'intfloat/e5-base-v2',
        'e5-large': 'intfloat/e5-large-v2',
        'gte-large': 'thenlper/gte-large',
        'jina-base': 'jinaai/jina-embeddings-v2-base-en',
        'finbert': 'yiyanghkust/finbert-tone',
        'legal-bert': 'nlpaueb/legal-bert-base-uncased',
        'law-embedding-1': 'law-ai/InLegalBERT',
        'pubmedbert': 'NeuML/pubmedbert-base-embeddings',
        'biobert': 'dmis-lab/biobert-base-cased-v1.2',
        'bioclinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
        'sapbert': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        'biomed-e5-large': 'BAAI/bge-large-en-v1.5'
    }

    def __init__(
            self,
            dense_model: str = 'minilm',
            sparse_method: str = 'bm25',
            alpha: float = 0.5,
            rrf_k: int = 60,
            verbose: bool = False
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_model: Name of dense embedding model
            sparse_method: 'bm25' or 'tfidf'
            alpha: Weight for dense retrieval (0-1), sparse gets (1-alpha)
            rrf_k: Constant for RRF formula (typically 60)
            verbose: Print debug info
        """
        self.dense_model_name = dense_model
        self.sparse_method = sparse_method
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.verbose = verbose

        # Initialize dense model
        model_path = self.EMBEDDING_MODELS.get(dense_model, dense_model)
        self.dense_model = SentenceTransformer(model_path)

        # Sparse components (initialized during indexing)
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.tokenized_docs = []

        # ChromaDB for dense retrieval
        self.chroma_client = None
        self.collection = None

        if verbose:
            print(f"HybridRetriever initialized:")
            print(f"  Dense model: {dense_model}")
            print(f"  Sparse method: {sparse_method}")
            print(f"  Alpha (dense weight): {alpha}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def index(self, documents: List[str]) -> None:
        """
        Index documents for both dense and sparse retrieval.

        Args:
            documents: List of document strings to index
        """
        if not documents:
            raise ValueError("No documents to index")

        self.documents = documents

        if self.verbose:
            print(f"Indexing {len(documents)} documents...")

        # === SPARSE INDEXING ===
        if self.sparse_method == 'bm25':
            self.tokenized_docs = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:  # tfidf
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # === DENSE INDEXING ===
        collection_id = hashlib.md5(
            f"{self.dense_model_name}_{len(documents)}_{documents[0][:30]}".encode()
        ).hexdigest()[:8]

        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        try:
            self.chroma_client.delete_collection(f"hybrid_{collection_id}")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=f"hybrid_{collection_id}",
            metadata={"hnsw:space": "cosine"}
        )

        embeddings = self.dense_model.encode(
            documents,
            show_progress_bar=self.verbose,
            convert_to_numpy=True
        ).tolist()

        self.collection.add(
            ids=[f"doc_{i}" for i in range(len(documents))],
            embeddings=embeddings,
            documents=documents
        )

        if self.verbose:
            print(f"Indexing complete. {len(documents)} documents indexed.")

    def _retrieve_dense(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense retrieval using embeddings."""
        query_embedding = self.dense_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k * 2, len(self.documents))
        )

        doc_scores = []
        for i, doc_id in enumerate(results['ids'][0]):
            doc_idx = int(doc_id.split('_')[1])
            distance = results['distances'][0][i] if results['distances'] else 0
            score = 1 - distance
            doc_scores.append((doc_idx, score))

        return doc_scores

    def _retrieve_sparse(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Sparse retrieval using BM25 or TF-IDF."""
        if self.sparse_method == 'bm25':
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
        else:
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        doc_scores = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return doc_scores

    def _reciprocal_rank_fusion(
            self,
            dense_results: List[Tuple[int, float]],
            sparse_results: List[Tuple[int, float]],
            top_k: int
    ) -> List[Tuple[int, float]]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = defaultdict(float)

        for rank, (doc_idx, _) in enumerate(dense_results):
            rrf_scores[doc_idx] += self.alpha * (1.0 / (self.rrf_k + rank + 1))

        for rank, (doc_idx, _) in enumerate(sparse_results):
            rrf_scores[doc_idx] += (1 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _linear_fusion(
            self,
            dense_results: List[Tuple[int, float]],
            sparse_results: List[Tuple[int, float]],
            top_k: int
    ) -> List[Tuple[int, float]]:
        """Combine results using linear score combination."""

        def normalize(results):
            if not results:
                return {}
            scores = [s for _, s in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {idx: 1.0 for idx, _ in results}
            return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}

        dense_norm = normalize(dense_results)
        sparse_norm = normalize(sparse_results)

        all_docs = set(dense_norm.keys()) | set(sparse_norm.keys())
        combined = {}

        for doc_idx in all_docs:
            dense_score = dense_norm.get(doc_idx, 0)
            sparse_score = sparse_norm.get(doc_idx, 0)
            combined[doc_idx] = self.alpha * dense_score + (1 - self.alpha) * sparse_score

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def retrieve(
            self,
            query: str,
            top_k: int = 5,
            fusion_method: str = 'rrf'
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid approach.

        Args:
            query: Search query
            top_k: Number of results to return
            fusion_method: 'rrf' or 'linear'

        Returns:
            List of RetrievalResult objects
        """
        if not self.documents:
            raise ValueError("No documents indexed. Call index() first.")

        dense_results = self._retrieve_dense(query, top_k)
        sparse_results = self._retrieve_sparse(query, top_k)

        if fusion_method == 'rrf':
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        else:
            fused_results = self._linear_fusion(dense_results, sparse_results, top_k)

        results = []
        for rank, (doc_idx, score) in enumerate(fused_results):
            results.append(RetrievalResult(
                content=self.documents[doc_idx],
                score=score,
                rank=rank + 1,
                source='hybrid'
            ))

        return results


def test_hybrid_retriever():
    """Test the hybrid retriever."""
    print("=" * 60)
    print("HYBRID RETRIEVER TEST")
    print("=" * 60)

    documents = [
        "Apple Inc. reported quarterly revenue of $89.5 billion, a 2% increase year over year.",
        "The Federal Reserve raised interest rates by 25 basis points to combat inflation.",
        "Tesla delivered 422,875 vehicles in Q1 2023, missing analyst expectations.",
        "Microsoft announced a $10 billion investment in OpenAI for artificial intelligence.",
        "Amazon Web Services revenue grew 16% to $21.4 billion in the quarter.",
        "Google's parent company Alphabet reported a decline in advertising revenue.",
        "JPMorgan Chase reported record profits driven by higher interest rates.",
        "Netflix added 7.66 million subscribers, exceeding market expectations.",
    ]

    retriever = HybridRetriever(
        dense_model='minilm',
        sparse_method='bm25',
        alpha=0.5,
        verbose=True
    )

    retriever.index(documents)

    queries = [
        "What was Apple's revenue?",
        "Federal Reserve interest rate decision",
        "cloud computing growth"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        results = retriever.retrieve(query, top_k=3)

        for r in results:
            print(f"  [{r.rank}] Score: {r.score:.4f}")
            print(f"      {r.content[:80]}...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_retriever()