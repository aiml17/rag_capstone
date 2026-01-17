"""
Multi-Embedding Retriever Module with Dense and Sparse Search
Supports:
- Dense retrieval: Embedding-based semantic search (MiniLM, MPNet, BGE, E5, FinBERT)
- Sparse retrieval: Lexical search (BM25, TF-IDF)

This aligns with RAGBench paper evaluation methodology (Figure 4a).
"""

import os
import sys
import re
import math
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Disable ChromaDB telemetry before importing
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TRUST_REMOTE_CODE"] = "True"

import logging
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.WARNING)

# Suppress telemetry error messages by patching
class SuppressTelemetry:
    def capture(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['posthog'] = SuppressTelemetry()

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import rank_bm25 for optimized BM25
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dense Embedding Models
# Replace EMBEDDING_MODELS in src/retriever.py with this:

EMBEDDING_MODELS = {
    # ============================================
    # GENERAL PURPOSE
    # ============================================
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Lightweight general-purpose model (baseline)",
        "type": "dense"
    },
    "mpnet": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "description": "Best general-purpose sentence transformer",
        "type": "dense"
    },

    # ============================================
    # BGE FAMILY
    # ============================================
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "description": "High-performance general encoder from BAAI",
        "type": "dense"
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "description": "Large BGE model, best retrieval performance",
        "type": "dense"
    },

    # ============================================
    # E5 FAMILY
    # ============================================
    "e5-base": {
        "model_name": "intfloat/e5-base-v2",
        "dimension": 768,
        "description": "Microsoft E5 model, strong on retrieval tasks",
        "type": "dense"
    },
    "e5-large": {
        "model_name": "intfloat/e5-large-v2",
        "dimension": 1024,
        "description": "Large E5 model, excellent retrieval quality",
        "type": "dense"
    },

    # ============================================
    # GTE FAMILY
    # ============================================
    "gte-large": {
        "model_name": "thenlper/gte-large",
        "dimension": 1024,
        "description": "General Text Embeddings large model",
        "type": "dense"
    },

    # ============================================
    # JINA
    # ============================================
    "jina-base": {
        "model_name": "jinaai/jina-embeddings-v2-base-en",
        "dimension": 768,
        "description": "Jina embeddings optimized for retrieval",
        "type": "dense"
    },

    # ============================================
    # FINANCE DOMAIN
    # ============================================
    "finbert": {
        "model_name": "yiyanghkust/finbert-tone",
        "dimension": 768,
        "description": "FinBERT pre-trained on financial communications",
        "type": "dense"
    },

    # ============================================
    # LEGAL DOMAIN
    # ============================================
    "legal-bert": {
        "model_name": "nlpaueb/legal-bert-base-uncased",
        "dimension": 768,
        "description": "Legal-BERT pre-trained on legal documents",
        "type": "dense"
    },
    "law-embedding-1": {
        "model_name": "law-ai/InLegalBERT",
        "dimension": 768,
        "description": "InLegalBERT for Indian legal documents",
        "type": "dense"
    },

    # ============================================
    # BIOMEDICAL DOMAIN
    # ============================================
    "pubmedbert": {
        "model_name": "NeuML/pubmedbert-base-embeddings",
        "dimension": 768,
        "description": "PubMedBERT fine-tuned for embeddings",
        "type": "dense"
    },
    "biobert": {
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "dimension": 768,
        "description": "BioBERT pre-trained on PubMed abstracts",
        "type": "dense"
    },
    "bioclinicalbert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "dimension": 768,
        "description": "ClinicalBERT for clinical notes and EHRs",
        "type": "dense"
    },
    "sapbert": {
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "dimension": 768,
        "description": "SapBERT for biomedical entity linking",
        "type": "dense"
    },
    "biomed-e5-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "description": "Using BGE-large as biomed fallback",
        "type": "dense"
    },
}
# Sparse Retrieval Methods
SPARSE_METHODS = {
    "bm25": {
        "description": "BM25 (Best Matching 25) - Probabilistic lexical ranking",
        "type": "sparse"
    },
    "tfidf": {
        "description": "TF-IDF - Term Frequency-Inverse Document Frequency",
        "type": "sparse"
    }
}

# Combined retrieval options
ALL_RETRIEVAL_METHODS = {**EMBEDDING_MODELS, **SPARSE_METHODS}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str = "dense"  # 'dense' or 'sparse'


# ============================================================================
# BM25 IMPLEMENTATION (Fallback if rank_bm25 not installed)
# ============================================================================

class CustomBM25:
    """
    Custom BM25 implementation.
    BM25 is a probabilistic ranking function used for lexical retrieval.

    Parameters:
        k1: Term frequency saturation (1.2-2.0 typical)
        b: Length normalization (0.75 typical)
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)

        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0

        self.doc_freqs = defaultdict(int)
        self.term_freqs = []

        for doc in corpus:
            term_freq = defaultdict(int)
            unique_terms = set()

            for term in doc:
                term_freq[term] += 1
                unique_terms.add(term)

            self.term_freqs.append(term_freq)

            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: List[str]) -> np.ndarray:
        """Calculate BM25 scores for all documents."""
        scores = np.zeros(self.corpus_size)

        for term in query:
            if term not in self.idf:
                continue

            idf = self.idf[term]

            for doc_idx in range(self.corpus_size):
                tf = self.term_freqs[doc_idx].get(term, 0)

                if tf > 0:
                    doc_len = self.doc_lengths[doc_idx]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                    scores[doc_idx] += idf * (numerator / denominator)

        return scores


# ============================================================================
# SPARSE RETRIEVER CLASS
# ============================================================================

class SparseRetriever:
    """
    Sparse retriever supporting BM25 and TF-IDF.

    Sparse search uses lexical matching (keyword-based) rather than
    semantic similarity like dense embeddings.
    """

    def __init__(self, method: str = "bm25"):
        """
        Initialize sparse retriever.

        Args:
            method: "bm25" or "tfidf"
        """
        self.method = method.lower()
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        self.tokenized_docs = []

        # BM25 components
        self.bm25 = None

        # TF-IDF components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self.is_indexed = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with stopword removal."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)

        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def index_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Index documents for sparse retrieval.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
            metadata: Optional list of metadata dicts

        Returns:
            Indexing statistics
        """
        start_time = time.time()

        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        self.doc_metadata = metadata or [{} for _ in documents]

        if self.method == "bm25":
            self._index_bm25()
        elif self.method == "tfidf":
            self._index_tfidf()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_indexed = True
        elapsed = time.time() - start_time

        return {
            "method": self.method,
            "documents_indexed": len(documents),
            "time_seconds": elapsed
        }

    def _index_bm25(self):
        """Index for BM25 retrieval."""
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]

        if HAS_RANK_BM25:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = CustomBM25(self.tokenized_docs)

    def _index_tfidf(self):
        """Index for TF-IDF retrieval."""
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filter

        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        if self.method == "bm25":
            return self._retrieve_bm25(query, top_k, filter_metadata)
        else:
            return self._retrieve_tfidf(query, top_k, filter_metadata)

    def _retrieve_bm25(self, query: str, top_k: int,
                       filter_metadata: Optional[Dict]) -> List[RetrievalResult]:
        """Retrieve using BM25."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filter
        if filter_metadata:
            for i, meta in enumerate(self.doc_metadata):
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    scores[i] = -1

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    chunk_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    metadata=self.doc_metadata[idx],
                    score=float(scores[idx]),
                    retrieval_method="bm25"
                ))

        return results

    def _retrieve_tfidf(self, query: str, top_k: int,
                        filter_metadata: Optional[Dict]) -> List[RetrievalResult]:
        """Retrieve using TF-IDF."""
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply metadata filter
        if filter_metadata:
            for i, meta in enumerate(self.doc_metadata):
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    scores[i] = -1

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    chunk_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    metadata=self.doc_metadata[idx],
                    score=float(scores[idx]),
                    retrieval_method="tfidf"
                ))

        return results


# ============================================================================
# DENSE RETRIEVER CLASS (Original MultiEmbeddingRetriever)
# ============================================================================

class MultiEmbeddingRetriever:
    """
    Dense retriever supporting multiple embedding models.
    Uses ChromaDB for vector storage and similarity search.
    """

    def __init__(
        self,
        embedding_key: str = "minilm",
        persist_directory: str = "./chroma_db",
        collection_prefix: str = "finqa",
        device: str = "mps"
    ):
        if embedding_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {embedding_key}. "
                           f"Available: {list(EMBEDDING_MODELS.keys())}")

        self.embedding_key = embedding_key
        self.model_config = EMBEDDING_MODELS[embedding_key]
        self.persist_directory = persist_directory
        self.collection_prefix = collection_prefix

        print(f"Loading embedding model: {self.model_config['model_name']}...")
        self.embedding_model = SentenceTransformer(
            self.model_config['model_name'],
            device=device
        )
        print(f"Model loaded. Embedding dimension: {self.model_config['dimension']}")

        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        self.collection_name = f"{collection_prefix}_{embedding_key}"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Collection '{self.collection_name}' ready. Documents: {self.collection.count()}")

    def _generate_chunk_id(self, content: str, doc_id: str, chunk_index: int) -> str:
        hash_input = f"{doc_id}_{chunk_index}_{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def index_documents(
        self,
        chunks: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """Index document chunks into the vector store."""
        if clear_existing and self.collection.count() > 0:
            print(f"Clearing existing {self.collection.count()} documents...")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        if metadatas is None:
            metadatas = [{"doc_id": doc_id} for doc_id in doc_ids]

        chunk_ids = [
            self._generate_chunk_id(chunk, doc_id, idx)
            for idx, (chunk, doc_id) in enumerate(zip(chunks, doc_ids))
        ]

        start_time = time.time()
        total_indexed = 0
        duplicates_skipped = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            existing = set(self.collection.get(ids=batch_ids)['ids'])

            new_chunks, new_ids, new_meta = [], [], []
            for chunk, cid, meta in zip(batch_chunks, batch_ids, batch_meta):
                if cid not in existing:
                    new_chunks.append(chunk)
                    new_ids.append(cid)
                    new_meta.append(meta)
                else:
                    duplicates_skipped += 1

            if new_chunks:
                embeddings = self.embedding_model.encode(
                    new_chunks, show_progress_bar=False, convert_to_numpy=True
                ).tolist()

                self.collection.add(
                    ids=new_ids,
                    embeddings=embeddings,
                    documents=new_chunks,
                    metadatas=new_meta
                )
                total_indexed += len(new_chunks)

            processed = min(i + batch_size, len(chunks))
            print(f"Progress: {processed}/{len(chunks)} chunks processed", end="\r")

        elapsed = time.time() - start_time
        print(f"\nIndexing complete in {elapsed:.2f}s")

        return {
            "total_chunks": len(chunks),
            "indexed": total_indexed,
            "duplicates_skipped": duplicates_skipped,
            "collection_size": self.collection.count(),
            "time_seconds": elapsed,
            "embedding_model": self.embedding_key,
            "retrieval_method": "dense"
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embedding_model.encode(
            query, convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        retrieval_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            similarity = 1 - distance

            retrieval_results.append(RetrievalResult(
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=similarity,
                retrieval_method="dense"
            ))

        return retrieval_results

    def retrieve_with_timing(self, query: str, top_k: int = 5) -> Tuple[List[RetrievalResult], float]:
        """Retrieve with timing information."""
        start = time.time()
        results = self.retrieve(query, top_k)
        return results, time.time() - start

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "embedding_model": self.embedding_key,
            "model_name": self.model_config['model_name'],
            "dimension": self.model_config['dimension'],
            "description": self.model_config['description'],
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "retrieval_method": "dense"
        }


# ============================================================================
# UNIFIED RETRIEVER CLASS
# ============================================================================

class UnifiedRetriever:
    """
    Unified retriever supporting both dense and sparse search methods.

    This class provides a single interface to:
    - Dense retrieval: Embedding-based (MiniLM, MPNet, BGE, E5, FinBERT)
    - Sparse retrieval: Lexical (BM25, TF-IDF)

    Usage:
        # Dense retrieval
        retriever = UnifiedRetriever(method="minilm")

        # Sparse retrieval
        retriever = UnifiedRetriever(method="bm25")
        retriever = UnifiedRetriever(method="tfidf")
    """

    def __init__(
        self,
        method: str = "minilm",
        persist_directory: str = "./chroma_db",
        collection_prefix: str = "finqa",
        device: str = "mps"
    ):
        """
        Initialize unified retriever.

        Args:
            method: Retrieval method - one of:
                    Dense: 'minilm', 'mpnet', 'bge-base', 'e5-base', 'finbert'
                    Sparse: 'bm25', 'tfidf'
            persist_directory: Directory for ChromaDB (dense only)
            collection_prefix: Prefix for collection names (dense only)
            device: Device for embeddings ('mps', 'cuda', 'cpu')
        """
        self.method = method.lower()
        self.is_dense = method in EMBEDDING_MODELS
        self.is_sparse = method in SPARSE_METHODS

        if not self.is_dense and not self.is_sparse:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Dense options: {list(EMBEDDING_MODELS.keys())}. "
                f"Sparse options: {list(SPARSE_METHODS.keys())}"
            )

        if self.is_dense:
            self.retriever = MultiEmbeddingRetriever(
                embedding_key=method,
                persist_directory=persist_directory,
                collection_prefix=collection_prefix,
                device=device
            )
            self.retrieval_type = "dense"
            print(f"Initialized DENSE retriever with {method}")
        else:
            self.retriever = SparseRetriever(method=method)
            self.retrieval_type = "sparse"
            print(f"Initialized SPARSE retriever with {method}")

    def index_documents(
        self,
        chunks: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Index documents for retrieval.

        Args:
            chunks: List of text chunks
            doc_ids: Optional document IDs
            metadatas: Optional metadata for each chunk
            **kwargs: Additional arguments (batch_size, clear_existing for dense)

        Returns:
            Indexing statistics
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(chunks))]

        if self.is_dense:
            return self.retriever.index_documents(
                chunks=chunks,
                doc_ids=doc_ids,
                metadatas=metadatas,
                **kwargs
            )
        else:
            return self.retriever.index_documents(
                documents=chunks,
                doc_ids=doc_ids,
                metadata=metadatas
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of RetrievalResult objects
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

    def retrieve_with_timing(self, query: str, top_k: int = 5) -> Tuple[List[RetrievalResult], float]:
        """Retrieve with timing information."""
        start = time.time()
        results = self.retrieve(query, top_k)
        return results, time.time() - start

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        if self.is_dense:
            return self.retriever.get_collection_stats()
        else:
            return {
                "method": self.method,
                "retrieval_type": "sparse",
                "documents_indexed": len(self.retriever.documents),
                "description": SPARSE_METHODS[self.method]["description"]
            }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_available_methods() -> Dict[str, Dict[str, Any]]:
    """Return information about all available retrieval methods."""
    return {
        "dense": EMBEDDING_MODELS.copy(),
        "sparse": SPARSE_METHODS.copy()
    }


def create_retriever(
    method: str,
    persist_directory: str = "./chroma_db",
    collection_prefix: str = "finqa",
    device: str = "mps"
) -> UnifiedRetriever:
    """
    Factory function to create a unified retriever.

    Args:
        method: 'minilm', 'mpnet', 'bge-base', 'e5-base', 'finbert', 'bm25', or 'tfidf'
        persist_directory: ChromaDB directory
        collection_prefix: Collection prefix
        device: Device for embeddings

    Returns:
        UnifiedRetriever instance
    """
    return UnifiedRetriever(
        method=method,
        persist_directory=persist_directory,
        collection_prefix=collection_prefix,
        device=device
    )


# Backward compatibility aliases
def create_dense_retriever(embedding_key: str, **kwargs) -> MultiEmbeddingRetriever:
    """Create a dense retriever (backward compatibility)."""
    return MultiEmbeddingRetriever(embedding_key=embedding_key, **kwargs)


def create_sparse_retriever(method: str = "bm25") -> SparseRetriever:
    """Create a sparse retriever."""
    return SparseRetriever(method=method)


# ============================================================================
# MAIN - Testing
# ============================================================================

def main():
    """Test both dense and sparse retrieval."""

    print("="*70)
    print("UNIFIED RETRIEVER TEST - DENSE vs SPARSE")
    print("="*70)

    # Sample financial documents
    documents = [
        "The company reported total revenue of $5.2 billion for fiscal year 2023, representing a 12% increase.",
        "Net income reached $890 million, up from $720 million in 2022, reflecting improved efficiency.",
        "The board approved a quarterly dividend of $0.50 per share, payable on March 15, 2024.",
        "Operating expenses decreased by 3% to $2.1 billion due to cost optimization initiatives.",
        "Research and development spending increased by 25% to support new product development.",
        "Cash and cash equivalents stood at $1.8 billion at year end, providing strong liquidity.",
        "The company repurchased 5 million shares at an average price of $45 per share.",
        "Revenue growth was driven by strong performance in the enterprise software segment.",
        "Gross profit margin improved to 45% from 42% in the prior year.",
        "Management expects revenue growth of 8-10% for the coming fiscal year."
    ]

    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    query = "What was the company's revenue growth rate?"

    # Test Dense (MiniLM)
    print("\n" + "="*70)
    print("TEST 1: DENSE RETRIEVAL (MiniLM)")
    print("="*70)

    dense_retriever = UnifiedRetriever(method="minilm", persist_directory="./test_chroma")
    dense_retriever.index_documents(documents, doc_ids, clear_existing=True)

    results, timing = dense_retriever.retrieve_with_timing(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"Time: {timing*1000:.2f}ms")
    print("Results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. (score: {r.score:.4f}) {r.content[:70]}...")

    # Test Sparse (BM25)
    print("\n" + "="*70)
    print("TEST 2: SPARSE RETRIEVAL (BM25)")
    print("="*70)

    bm25_retriever = UnifiedRetriever(method="bm25")
    bm25_retriever.index_documents(documents, doc_ids)

    results, timing = bm25_retriever.retrieve_with_timing(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"Time: {timing*1000:.2f}ms")
    print("Results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. (score: {r.score:.4f}) {r.content[:70]}...")

    # Test Sparse (TF-IDF)
    print("\n" + "="*70)
    print("TEST 3: SPARSE RETRIEVAL (TF-IDF)")
    print("="*70)

    tfidf_retriever = UnifiedRetriever(method="tfidf")
    tfidf_retriever.index_documents(documents, doc_ids)

    results, timing = tfidf_retriever.retrieve_with_timing(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"Time: {timing*1000:.2f}ms")
    print("Results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. (score: {r.score:.4f}) {r.content[:70]}...")

    # Show available methods
    print("\n" + "="*70)
    print("AVAILABLE RETRIEVAL METHODS")
    print("="*70)
    methods = get_available_methods()
    print("\nDense (Embedding-based):")
    for key, info in methods['dense'].items():
        print(f"  - {key}: {info['description']}")
    print("\nSparse (Lexical):")
    for key, info in methods['sparse'].items():
        print(f"  - {key}: {info['description']}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()