"""
Enhanced RAG Pipeline for FinQA Dataset
Compares and selects best:
1. Embedding models
2. Chunking strategies
3. LLM models for generation
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Disable telemetry before imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

# Suppress telemetry warnings
class TelemetryFilter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, msg):
        if 'telemetry' not in msg.lower() and 'capture()' not in msg:
            self.stream.write(msg)
    def flush(self):
        self.stream.flush()

sys.stderr = TelemetryFilter(sys.__stderr__)

from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import DocumentChunker
from src.retriever import MultiEmbeddingRetriever, EMBEDDING_MODELS
from src.generator import RAGGenerator


@dataclass
class EmbeddingBenchmarkResult:
    """Results from embedding model comparison."""
    model_key: str
    model_name: str
    avg_retrieval_score: float
    avg_retrieval_time_ms: float
    num_examples: int


@dataclass
class ChunkingBenchmarkResult:
    """Results from chunking strategy comparison."""
    strategy: str
    params: Dict[str, Any]
    avg_num_chunks: float
    avg_retrieval_score: float
    num_examples: int


@dataclass
class LLMBenchmarkResult:
    """Results from LLM model comparison."""
    model_key: str
    model_name: str
    avg_semantic_similarity: float
    avg_generation_time_ms: float
    avg_tokens_used: float
    num_examples: int


@dataclass
class PipelineConfig:
    """Optimal pipeline configuration."""
    embedding_model: str
    chunking_strategy: str
    chunking_params: Dict[str, Any]
    llm_model: str
    retrieval_score: float
    semantic_similarity: float


class EnhancedRAGPipeline:
    """
    Enhanced RAG Pipeline with comprehensive comparison capabilities.
    Finds optimal configuration for embedding, chunking, and generation.
    """

    # LLM models to compare
    LLM_MODELS = {
        'llama-3.1-8b': 'Fast, good quality',
        'llama-3.3-70b': 'Best quality, slower',
        'qwen3-32b': 'Alibaba Qwen, good reasoning'
    }

    # Chunking strategies to compare
    CHUNKING_STRATEGIES = {
        'none': {},
        'sentence': {'chunk_size': 5, 'overlap': 2},
        'semantic': {'similarity_threshold': 0.5}
    }

    def __init__(
        self,
        persist_directory: str = "./chroma_db_pipeline",
        device: Optional[str] = None
    ):
        """Initialize the enhanced pipeline."""
        if device is None:
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = device
        self.persist_directory = persist_directory
        self.dataset = None

        # Results storage
        self.embedding_results: List[EmbeddingBenchmarkResult] = []
        self.chunking_results: List[ChunkingBenchmarkResult] = []
        self.llm_results: List[LLMBenchmarkResult] = []
        self.optimal_config: Optional[PipelineConfig] = None

        print(f"Enhanced RAG Pipeline initialized on device: {device}")

    def load_data(self, split: str = "test", num_samples: Optional[int] = None):
        """Load the FinQA dataset."""
        print("\nLoading RAGBench FinQA dataset...")
        self.dataset = load_dataset("rungalileo/ragbench", "finqa", split=split)

        if num_samples:
            indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
            self.dataset = self.dataset.select(indices)

        print(f"Loaded {len(self.dataset)} examples")
        return self.dataset

    def _get_documents(self, example: Dict) -> List[str]:
        """Extract documents from example."""
        documents = example.get('documents', example.get('context', []))
        if isinstance(documents, str):
            documents = [documents]
        return [d for d in documents if isinstance(d, str) and d.strip()]

    # =========================================================================
    # PHASE 1: EMBEDDING MODEL COMPARISON
    # =========================================================================

    def compare_embedding_models(
        self,
        models_to_test: Optional[List[str]] = None,
        num_examples: int = 20,
        top_k: int = 3
    ) -> List[EmbeddingBenchmarkResult]:
        """
        Compare different embedding models on retrieval quality.

        Args:
            models_to_test: List of model keys from EMBEDDING_MODELS
            num_examples: Number of examples to test
            top_k: Number of chunks to retrieve

        Returns:
            List of benchmark results sorted by retrieval score
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_data() first.")

        if models_to_test is None:
            models_to_test = list(EMBEDDING_MODELS.keys())

        print("\n" + "=" * 70)
        print("PHASE 1: EMBEDDING MODEL COMPARISON")
        print("=" * 70)

        # Use semantic chunking for embedding comparison
        chunker = DocumentChunker(device=self.device)
        samples = [self.dataset[i] for i in range(min(num_examples, len(self.dataset)))]

        self.embedding_results = []

        for model_key in models_to_test:
            print(f"\n--- Testing: {model_key} ---")
            model_config = EMBEDDING_MODELS[model_key]

            try:
                retriever = MultiEmbeddingRetriever(
                    embedding_key=model_key,
                    persist_directory=self.persist_directory,
                    collection_prefix="embed_compare"
                )
            except Exception as e:
                print(f"Error loading {model_key}: {e}")
                continue

            retrieval_scores = []
            retrieval_times = []

            for example in tqdm(samples, desc=f"Evaluating {model_key}"):
                documents = self._get_documents(example)
                if not documents:
                    continue

                # Chunk documents
                chunks = chunker.chunk_documents(documents, strategy='semantic')
                if not chunks:
                    continue

                chunk_texts = [c['text'] for c in chunks]
                doc_ids = [f"doc_{c['doc_idx']}" for c in chunks]

                # Index
                retriever.index_documents(
                    chunks=chunk_texts,
                    doc_ids=doc_ids,
                    clear_existing=True
                )

                # Retrieve with timing
                question = example.get('question', '')
                results, elapsed = retriever.retrieve_with_timing(question, top_k)

                if results:
                    avg_score = np.mean([r.score for r in results])
                    retrieval_scores.append(avg_score)
                    retrieval_times.append(elapsed * 1000)

            if retrieval_scores:
                result = EmbeddingBenchmarkResult(
                    model_key=model_key,
                    model_name=model_config['model_name'],
                    avg_retrieval_score=np.mean(retrieval_scores),
                    avg_retrieval_time_ms=np.mean(retrieval_times),
                    num_examples=len(retrieval_scores)
                )
                self.embedding_results.append(result)
                print(f"  Score: {result.avg_retrieval_score:.4f}, Time: {result.avg_retrieval_time_ms:.2f}ms")

        # Sort by score
        self.embedding_results.sort(key=lambda x: x.avg_retrieval_score, reverse=True)
        return self.embedding_results

    # =========================================================================
    # PHASE 2: CHUNKING STRATEGY COMPARISON
    # =========================================================================

    def compare_chunking_strategies(
        self,
        embedding_model: str = "minilm",
        num_examples: int = 20,
        top_k: int = 3
    ) -> List[ChunkingBenchmarkResult]:
        """
        Compare different chunking strategies using the specified embedding model.

        Args:
            embedding_model: Embedding model to use
            num_examples: Number of examples to test
            top_k: Number of chunks to retrieve

        Returns:
            List of benchmark results sorted by retrieval score
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_data() first.")

        print("\n" + "=" * 70)
        print(f"PHASE 2: CHUNKING STRATEGY COMPARISON (using {embedding_model})")
        print("=" * 70)

        chunker = DocumentChunker(device=self.device)
        retriever = MultiEmbeddingRetriever(
            embedding_key=embedding_model,
            persist_directory=self.persist_directory,
            collection_prefix="chunk_compare"
        )

        samples = [self.dataset[i] for i in range(min(num_examples, len(self.dataset)))]
        self.chunking_results = []

        for strategy_name, params in self.CHUNKING_STRATEGIES.items():
            print(f"\n--- Testing: {strategy_name} ---")

            retrieval_scores = []
            chunk_counts = []

            for example in tqdm(samples, desc=f"Evaluating {strategy_name}"):
                documents = self._get_documents(example)
                if not documents:
                    continue

                # Apply chunking strategy
                if strategy_name == 'none':
                    chunks = [{'text': doc, 'doc_idx': i} for i, doc in enumerate(documents)]
                else:
                    chunks = chunker.chunk_documents(documents, strategy=strategy_name, **params)

                if not chunks:
                    continue

                chunk_texts = [c['text'] for c in chunks]
                doc_ids = [f"doc_{c.get('doc_idx', 0)}" for c in chunks]
                chunk_counts.append(len(chunks))

                # Index and retrieve
                retriever.index_documents(chunks=chunk_texts, doc_ids=doc_ids, clear_existing=True)

                question = example.get('question', '')
                results = retriever.retrieve(question, top_k)

                if results:
                    avg_score = np.mean([r.score for r in results])
                    retrieval_scores.append(avg_score)

            if retrieval_scores:
                result = ChunkingBenchmarkResult(
                    strategy=strategy_name,
                    params=params,
                    avg_num_chunks=np.mean(chunk_counts),
                    avg_retrieval_score=np.mean(retrieval_scores),
                    num_examples=len(retrieval_scores)
                )
                self.chunking_results.append(result)
                print(f"  Score: {result.avg_retrieval_score:.4f}, Avg Chunks: {result.avg_num_chunks:.1f}")

        # Sort by score
        self.chunking_results.sort(key=lambda x: x.avg_retrieval_score, reverse=True)
        return self.chunking_results

    # =========================================================================
    # PHASE 3: LLM MODEL COMPARISON
    # =========================================================================

    def compare_llm_models(
        self,
        embedding_model: str = "minilm",
        chunking_strategy: str = "semantic",
        chunking_params: Optional[Dict] = None,
        models_to_test: Optional[List[str]] = None,
        num_examples: int = 10,
        top_k: int = 3
    ) -> List[LLMBenchmarkResult]:
        """
        Compare different LLM models on generation quality.

        Args:
            embedding_model: Embedding model to use for retrieval
            chunking_strategy: Chunking strategy to use
            chunking_params: Chunking parameters
            models_to_test: List of LLM model keys
            num_examples: Number of examples to test
            top_k: Number of chunks to retrieve

        Returns:
            List of benchmark results sorted by semantic similarity
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_data() first.")

        if models_to_test is None:
            models_to_test = list(self.LLM_MODELS.keys())

        if chunking_params is None:
            chunking_params = self.CHUNKING_STRATEGIES.get(chunking_strategy, {})

        print("\n" + "=" * 70)
        print(f"PHASE 3: LLM MODEL COMPARISON")
        print(f"Using: {embedding_model} embeddings + {chunking_strategy} chunking")
        print("=" * 70)

        # Setup retrieval pipeline
        chunker = DocumentChunker(device=self.device)
        retriever = MultiEmbeddingRetriever(
            embedding_key=embedding_model,
            persist_directory=self.persist_directory,
            collection_prefix="llm_compare"
        )

        # Load a model for computing semantic similarity
        similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)

        samples = [self.dataset[i] for i in range(min(num_examples, len(self.dataset)))]
        self.llm_results = []

        for model_key in models_to_test:
            print(f"\n--- Testing: {model_key} ---")

            try:
                generator = RAGGenerator(model_name=model_key)
            except Exception as e:
                print(f"Error initializing {model_key}: {e}")
                continue

            similarities = []
            generation_times = []
            tokens_used = []

            for example in tqdm(samples, desc=f"Evaluating {model_key}"):
                documents = self._get_documents(example)
                question = example.get('question', '')
                ground_truth = example.get('response', example.get('answer', ''))

                if not documents or not ground_truth:
                    continue

                # Chunk and retrieve
                if chunking_strategy == 'none':
                    chunks = [{'text': doc, 'doc_idx': i} for i, doc in enumerate(documents)]
                else:
                    chunks = chunker.chunk_documents(documents, strategy=chunking_strategy, **chunking_params)

                if not chunks:
                    continue

                chunk_texts = [c['text'] for c in chunks]
                doc_ids = [f"doc_{c.get('doc_idx', 0)}" for c in chunks]

                retriever.index_documents(chunks=chunk_texts, doc_ids=doc_ids, clear_existing=True)
                results = retriever.retrieve(question, top_k)

                # Format for generator
                generator_chunks = [{'text': r.content, 'score': r.score} for r in results]

                # Generate with timing
                start_time = time.time()
                generation = generator.generate(question, generator_chunks)
                elapsed = (time.time() - start_time) * 1000

                generated_text = generation.get('response', '')

                if generated_text and 'error' not in generation:
                    # Compute semantic similarity
                    gt_embedding = similarity_model.encode([ground_truth])
                    gen_embedding = similarity_model.encode([generated_text])
                    sim = cosine_similarity(gt_embedding, gen_embedding)[0][0]

                    similarities.append(sim)
                    generation_times.append(elapsed)

                    usage = generation.get('usage', {})
                    if usage:
                        tokens_used.append(usage.get('total_tokens', 0))

            if similarities:
                result = LLMBenchmarkResult(
                    model_key=model_key,
                    model_name=generator.model,
                    avg_semantic_similarity=np.mean(similarities),
                    avg_generation_time_ms=np.mean(generation_times),
                    avg_tokens_used=np.mean(tokens_used) if tokens_used else 0,
                    num_examples=len(similarities)
                )
                self.llm_results.append(result)
                print(f"  Similarity: {result.avg_semantic_similarity:.4f}, Time: {result.avg_generation_time_ms:.0f}ms")

        # Sort by similarity
        self.llm_results.sort(key=lambda x: x.avg_semantic_similarity, reverse=True)
        return self.llm_results

    # =========================================================================
    # COMPREHENSIVE COMPARISON
    # =========================================================================

    def run_full_comparison(
        self,
        num_examples_embedding: int = 20,
        num_examples_chunking: int = 20,
        num_examples_llm: int = 10,
        embedding_models: Optional[List[str]] = None,
        llm_models: Optional[List[str]] = None
    ) -> PipelineConfig:
        """
        Run full comparison across all dimensions and return optimal configuration.

        Args:
            num_examples_embedding: Examples for embedding comparison
            num_examples_chunking: Examples for chunking comparison
            num_examples_llm: Examples for LLM comparison
            embedding_models: Embedding models to test (None = all)
            llm_models: LLM models to test (None = all)

        Returns:
            Optimal pipeline configuration
        """
        print("\n" + "=" * 70)
        print("FULL PIPELINE COMPARISON")
        print("=" * 70)

        # Phase 1: Embedding models
        self.compare_embedding_models(
            models_to_test=embedding_models,
            num_examples=num_examples_embedding
        )
        best_embedding = self.embedding_results[0].model_key if self.embedding_results else "minilm"

        # Phase 2: Chunking strategies (using best embedding)
        self.compare_chunking_strategies(
            embedding_model=best_embedding,
            num_examples=num_examples_chunking
        )
        best_chunking = self.chunking_results[0] if self.chunking_results else None
        chunking_strategy = best_chunking.strategy if best_chunking else "semantic"
        chunking_params = best_chunking.params if best_chunking else {}

        # Phase 3: LLM models (using best embedding + chunking)
        self.compare_llm_models(
            embedding_model=best_embedding,
            chunking_strategy=chunking_strategy,
            chunking_params=chunking_params,
            models_to_test=llm_models,
            num_examples=num_examples_llm
        )
        best_llm = self.llm_results[0].model_key if self.llm_results else "llama-3.1-8b"

        # Store optimal configuration
        self.optimal_config = PipelineConfig(
            embedding_model=best_embedding,
            chunking_strategy=chunking_strategy,
            chunking_params=chunking_params,
            llm_model=best_llm,
            retrieval_score=self.embedding_results[0].avg_retrieval_score if self.embedding_results else 0,
            semantic_similarity=self.llm_results[0].avg_semantic_similarity if self.llm_results else 0
        )

        return self.optimal_config

    def print_comparison_summary(self):
        """Print a comprehensive summary of all comparisons."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        # Embedding results
        if self.embedding_results:
            print("\n--- EMBEDDING MODELS ---")
            print(f"{'Model':<25} {'Score':<12} {'Time (ms)':<12}")
            print("-" * 50)
            for r in self.embedding_results:
                marker = " <-- BEST" if r == self.embedding_results[0] else ""
                print(f"{r.model_key:<25} {r.avg_retrieval_score:<12.4f} {r.avg_retrieval_time_ms:<12.2f}{marker}")

        # Chunking results
        if self.chunking_results:
            print("\n--- CHUNKING STRATEGIES ---")
            print(f"{'Strategy':<15} {'Score':<12} {'Avg Chunks':<12}")
            print("-" * 40)
            for r in self.chunking_results:
                marker = " <-- BEST" if r == self.chunking_results[0] else ""
                print(f"{r.strategy:<15} {r.avg_retrieval_score:<12.4f} {r.avg_num_chunks:<12.1f}{marker}")

        # LLM results
        if self.llm_results:
            print("\n--- LLM MODELS ---")
            print(f"{'Model':<20} {'Similarity':<12} {'Time (ms)':<12} {'Tokens':<10}")
            print("-" * 55)
            for r in self.llm_results:
                marker = " <-- BEST" if r == self.llm_results[0] else ""
                print(f"{r.model_key:<20} {r.avg_semantic_similarity:<12.4f} {r.avg_generation_time_ms:<12.0f} {r.avg_tokens_used:<10.0f}{marker}")

        # Optimal configuration
        if self.optimal_config:
            print("\n" + "=" * 70)
            print("OPTIMAL CONFIGURATION")
            print("=" * 70)
            print(f"  Embedding Model:    {self.optimal_config.embedding_model}")
            print(f"  Chunking Strategy:  {self.optimal_config.chunking_strategy}")
            print(f"  Chunking Params:    {self.optimal_config.chunking_params}")
            print(f"  LLM Model:          {self.optimal_config.llm_model}")
            print(f"  Retrieval Score:    {self.optimal_config.retrieval_score:.4f}")
            print(f"  Semantic Similarity: {self.optimal_config.semantic_similarity:.4f}")

    def save_results(self, filepath: str = "pipeline_comparison_results.json"):
        """Save all comparison results to JSON."""

        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        output = {
            "timestamp": datetime.now().isoformat(),
            "embedding_results": [convert_to_native(asdict(r)) for r in self.embedding_results],
            "chunking_results": [convert_to_native(asdict(r)) for r in self.chunking_results],
            "llm_results": [convert_to_native(asdict(r)) for r in self.llm_results],
            "optimal_config": convert_to_native(asdict(self.optimal_config)) if self.optimal_config else None
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filepath}")


def main():
    """Run the full pipeline comparison."""

    print("=" * 70)
    print("ENHANCED RAG PIPELINE - FULL COMPARISON")
    print("=" * 70)

    # Initialize pipeline
    pipeline = EnhancedRAGPipeline(persist_directory="./chroma_db_comparison")

    # Load data
    pipeline.load_data(split="test", num_samples=50)

    # Run full comparison
    # Using smaller sample sizes for faster testing - increase for production
    optimal_config = pipeline.run_full_comparison(
        num_examples_embedding=20,
        num_examples_chunking=20,
        num_examples_llm=10,
        embedding_models=["minilm", "mpnet", "bge-base", "e5-base", "finbert"],
        llm_models=["llama-3.1-8b", "llama-3.3-70b","qwen3-32b","gpt-oss-120b","compound"]
    )

    # Print summary
    pipeline.print_comparison_summary()

    # Save results
    pipeline.save_results()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()