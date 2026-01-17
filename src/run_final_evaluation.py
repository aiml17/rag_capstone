"""
Final RAG Pipeline Evaluation Script
Tests TOP 10 configurations with 100 examples each.

Features:
- Top 10 best-performing configurations
- 100 examples per configuration (1000 total evaluations)
- Checkpoint every 10 examples
- Auto-rotation of Groq API keys with validation
- Local Ollama judge (qwen2.5:32b)
- TRACe metrics (Relevance, Utilization, Completeness, Adherence)
- RMSE for Relevance, Utilization, Completeness (not Adherence per RAGBench methodology)
- AUROC, F1, Precision, Recall for Adherence classification
- Captures dataset IDs, generated responses, and retrieved chunks

Usage:
    python run_final_evaluation.py --run              # Run evaluation
    python run_final_evaluation.py --progress         # Show progress
    python run_final_evaluation.py --results          # Show current results
"""

import os
import sys
import json
import time
import shutil
import argparse
import uuid
import glob
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# ============================================================================
# DISABLE TELEMETRY
# ============================================================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import posthog
    posthog.disabled = True
    posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import RAGBenchLoader
from src.chunker import DocumentChunker
from src.evaluator import TRACeEvaluator
from src.retriever import UnifiedRetriever
from src.hybrid_retriever import HybridRetriever

from groq import Groq

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_EXAMPLES = 100          # 100 examples per configuration
CHECKPOINT_INTERVAL = 10    # Save checkpoint every 10 examples
RANDOM_SEED = 42
TOP_K = 5
DELAY_BETWEEN_CALLS = 2.0  # Delay between API calls (single key)

# Judge Configuration (Local Ollama)
JUDGE_MODEL = "qwen2.5:7b"
USE_LOCAL_JUDGE = True

# Retry Configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 10
BACKOFF_MULTIPLIER = 2

# Example-level retry configuration
MAX_EXAMPLE_RETRIES = 2          # Retry failed examples up to 2 times
RETRY_ON_ZERO_TRACE = True       # Retry if TRACe score is 0
MIN_VALID_TRACE = 0.01           # Minimum TRACe to consider valid
RETRY_DELAY = 5                  # Delay between example retries

# Directories
CHECKPOINT_DIR = "final_eval_checkpoints"
RESULTS_DIR = "final_eval_results"

# ============================================================================
# TOP 10 CONFIGURATIONS (Based on previous 142-config evaluation)
# Format: (config_type, chunking, retrieval_method, llm_model)
# ============================================================================

TOP_10_CONFIGS = [
    # Rank 1: Best overall TRACe (sparse tfidf)
    ("sparse", "semantic", "tfidf", "qwen/qwen3-32b"),

    # Rank 2: Second best sparse (bm25)
    ("sparse", "semantic", "bm25", "qwen/qwen3-32b"),

    # Rank 3: Best sparse with compound
    ("sparse", "semantic", "tfidf", "groq/compound"),

    # Rank 4: Best hybrid (finbert + tfidf)
    ("hybrid", "semantic", "finbert", "groq/compound"),

    # Rank 5: Best dense with finbert
    ("dense", "semantic", "finbert", "qwen/qwen3-32b"),

    # Rank 6: Dense with BGE
    ("dense", "semantic", "bge-base", "groq/compound"),

    # Rank 7: Dense with MiniLM (lowest RMSE config)
    ("dense", "semantic", "minilm", "openai/gpt-oss-120b"),

    # Rank 8: Sparse BM25 with compound
    ("sparse", "semantic", "bm25", "groq/compound"),

    # Rank 9: Dense with MPNet
    ("dense", "semantic", "mpnet", "qwen/qwen3-32b"),

    # Rank 10: Dense finbert with compound
    ("dense", "semantic", "finbert", "groq/compound"),
]

# ============================================================================
# API KEY SETUP (Single Key - No Rotation)
# ============================================================================

def get_api_key() -> str:
    """Get single API key from environment."""
    for i in range(1, 21):
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key:
            return key
    key = os.getenv('GROQ_API_KEY')
    if key:
        return key
    print("❌ No API key found! Set GROQ_API_KEY in .env")
    sys.exit(1)


def validate_api_key(key: str) -> Tuple[bool, str]:
    """Test if an API key is valid."""
    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        return True, "OK"
    except Exception as e:
        return False, str(e)[:100]


class RateLimitExit(Exception):
    """Exception raised when rate limit is hit to trigger graceful exit."""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_temp_folders():
    """Remove temporary ChromaDB folders."""
    for pattern in ['temp_chroma_*', 'chroma_db_eval_*']:
        for folder in glob.glob(pattern):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExampleResult:
    """Result for a single example."""
    example_idx: int
    dataset_id: str                          # RAGBench dataset ID
    question: str                            # Original question
    ground_truth_answer: str                 # Ground truth answer from dataset
    generated_response: str                  # LLM generated response
    retrieved_chunks: List[str]              # Retrieved chunks/sentences
    retrieved_chunk_ids: List[str]           # IDs/keys of retrieved chunks
    computed_relevance: float
    computed_utilization: float
    computed_completeness: float
    computed_adherence: float
    gt_relevance: float
    gt_utilization: float
    gt_completeness: float
    gt_adherence: float
    binary_pred: int
    binary_gt: int
    is_valid: bool = True
    error: str = ""


@dataclass
class ConfigResult:
    """Aggregated results for a configuration."""
    config_type: str
    chunking_strategy: str
    retrieval_method: str
    llm_model: str

    # TRACe averages
    avg_relevance: float = 0.0
    avg_utilization: float = 0.0
    avg_completeness: float = 0.0
    avg_adherence: float = 0.0

    # Ground truth averages
    gt_avg_relevance: float = 0.0
    gt_avg_utilization: float = 0.0
    gt_avg_completeness: float = 0.0
    gt_avg_adherence: float = 0.0

    # RMSE scores (Relevance, Utilization, Completeness only - not Adherence)
    rmse_relevance: float = 0.0
    rmse_utilization: float = 0.0
    rmse_completeness: float = 0.0

    # Classification metrics
    auroc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Metadata
    num_examples: int = 0
    num_valid: int = 0
    num_failed: int = 0
    example_results: List[Dict] = field(default_factory=list)


# ============================================================================
# MAIN EVALUATOR CLASS
# ============================================================================

class FinalEvaluator:
    """Evaluator for top 10 configurations with 100 examples each."""

    def __init__(self):
        self.samples = []
        self.sample_ids = []  # Store dataset IDs
        self.chunker = DocumentChunker()
        self.trace_evaluator = None
        self.groq_client = None
        self.api_key = None

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        cleanup_temp_folders()
        print("✓ Cleaned up temp folders")

    def load_data(self):
        """Load 100 samples from RAGBench FinQA."""
        print("\n📊 Loading RAGBench FinQA data...")
        loader = RAGBenchLoader()
        all_samples = loader.load_finqa(split="test")

        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(all_samples), size=NUM_EXAMPLES, replace=False)
        indices = [int(i) for i in sorted(indices)]

        self.samples = [all_samples[i] for i in indices]

        # Extract dataset IDs - try multiple possible field names
        self.sample_ids = []
        for i, sample in zip(indices, self.samples):
            # Try various ID field names that RAGBench might use
            dataset_id = (
                sample.get('id') or
                sample.get('idx') or
                sample.get('example_id') or
                sample.get('qid') or
                sample.get('question_id') or
                f"finqa_test_{i}"  # Fallback: use index
            )
            self.sample_ids.append(str(dataset_id))

        print(f"✓ Loaded {len(self.samples)} samples (seed={RANDOM_SEED})")
        print(f"  Sample IDs: {self.sample_ids[:3]}... (showing first 3)")

    def initialize_judge(self):
        """Initialize the TRACe evaluator."""
        print(f"\n🧑‍⚖️ Initializing judge: {JUDGE_MODEL}")
        self.trace_evaluator = TRACeEvaluator(
            judge_model=JUDGE_MODEL,
            use_local=USE_LOCAL_JUDGE
        )

    def initialize_api_key(self):
        """Initialize and validate single API key."""
        self.api_key = get_api_key()
        key_preview = f"{self.api_key[:8]}...{self.api_key[-4:]}"

        print(f"\n🔑 Validating API key ({key_preview})...")
        is_valid, msg = validate_api_key(self.api_key)

        if is_valid:
            print(f"  ✓ API key is valid")
            self.groq_client = Groq(api_key=self.api_key)
        else:
            print(f"  ✗ API key invalid: {msg}")
            sys.exit(1)

    def _get_config_id(self, config: Tuple) -> str:
        """Generate unique ID for a configuration."""
        return f"{config[0]}_{config[1]}_{config[2]}_{config[3].split('/')[-1]}"

    def _load_checkpoint(self, config_id: str) -> Dict:
        """Load checkpoint for a configuration."""
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{config_id}.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed_examples": [], "example_results": []}

    def _save_checkpoint(self, config_id: str, data: Dict):
        """Save checkpoint for a configuration."""
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{config_id}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_response(self, question: str, context: str, llm_model: str) -> str:
        """Generate response using Groq API. Raises RateLimitExit on rate limit."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Provide a clear, accurate answer based only on the information in the context."""

        try:
            response = self.groq_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                raise RateLimitExit(f"Rate limit hit: {str(e)[:100]}")
            elif 'organization' in error_str and 'restricted' in error_str:
                raise RateLimitExit(f"Organization restricted: {str(e)[:100]}")
            else:
                raise e

    def _create_retriever(self, config: Tuple, corpus_chunks: List[str]):
        """Create retriever based on configuration."""
        config_type, chunking, method, llm = config

        if config_type == "dense":
            temp_dir = f"temp_chroma_{method}_{uuid.uuid4().hex[:8]}"
            retriever = UnifiedRetriever(
                method=method,
                persist_directory=temp_dir,
                collection_prefix="eval"
            )
            retriever.index_documents(
                chunks=corpus_chunks,
                doc_ids=[f"chunk_{i}" for i in range(len(corpus_chunks))],
                clear_existing=True
            )
            return retriever

        elif config_type == "sparse":
            retriever = UnifiedRetriever(method=method)
            retriever.index_documents(
                chunks=corpus_chunks,
                doc_ids=[f"chunk_{i}" for i in range(len(corpus_chunks))]
            )
            return retriever

        elif config_type == "hybrid":
            retriever = HybridRetriever(
                dense_model=method,
                sparse_method='tfidf',
                alpha=0.7,
                verbose=False
            )
            retriever.index(corpus_chunks)
            return retriever

        raise ValueError(f"Unknown config type: {config_type}")

    def _retrieve(self, retriever, query: str, config_type: str) -> Tuple[List[str], List[str]]:
        """Retrieve documents using the appropriate method.
        Returns: (chunks, chunk_ids)
        """
        if config_type == "hybrid":
            results = retriever.retrieve(query, top_k=TOP_K)
            chunks = [r.content for r in results]
            # Try to get IDs from results, fallback to index-based IDs
            chunk_ids = []
            for i, r in enumerate(results):
                chunk_id = getattr(r, 'doc_id', None) or getattr(r, 'id', None) or f"chunk_{i}"
                chunk_ids.append(str(chunk_id))
            return chunks, chunk_ids
        else:
            results = retriever.retrieve(query, top_k=TOP_K)
            chunks = [r.content for r in results]
            chunk_ids = []
            for i, r in enumerate(results):
                chunk_id = getattr(r, 'doc_id', None) or getattr(r, 'id', None) or f"chunk_{i}"
                chunk_ids.append(str(chunk_id))
            return chunks, chunk_ids

    def evaluate_example(self, idx: int, sample: Dict, config: Tuple) -> ExampleResult:
        """Evaluate a single example."""
        config_type, chunking, method, llm_model = config

        # Get dataset ID
        dataset_id = self.sample_ids[idx]

        # Get question and ground truth
        question = sample.get('question', '')
        ground_truth_answer = sample.get('response', sample.get('answer', ''))

        # Build corpus
        documents = sample.get('documents', sample.get('context', []))
        if isinstance(documents, str):
            documents = [documents]
        corpus = "\n\n".join(documents)

        # Chunk
        if chunking == "none":
            chunks = [corpus] if corpus else []
        elif chunking == "sentence":
            chunk_results = self.chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
            chunks = [c['text'] for c in chunk_results]
        else:  # semantic
            chunk_results = self.chunker.semantic_chunking(corpus)
            chunks = [c['text'] for c in chunk_results]

        if not chunks:
            chunks = [corpus] if corpus else ["No content"]

        cleanup_temp_folders()

        # Create retriever and retrieve
        retriever = self._create_retriever(config, chunks)
        retrieved_chunks, retrieved_chunk_ids = self._retrieve(retriever, question, config_type)
        context = "\n\n".join(retrieved_chunks)

        cleanup_temp_folders()

        # Generate response
        generated_response = self._generate_response(question, context, llm_model)

        # Evaluate with TRACe
        result = self.trace_evaluator.evaluate_single(
            question=question,
            ground_truth=ground_truth_answer,
            generated_response=generated_response,
            retrieved_chunks=retrieved_chunks
        )

        trace_scores = result.trace_scores

        # Get ground truth scores from dataset
        gt_rel = float(sample.get('context_relevance', sample.get('relevance_score', 0.5)))
        gt_util = float(sample.get('context_utilization', sample.get('utilization_score', 0.5)))
        gt_comp = float(sample.get('completeness_score', sample.get('completeness', 0.5)))
        gt_adh = float(sample.get('adherence_score', 1.0 if sample.get('adherence', True) else 0.0))

        # Binary for AUROC/F1
        comp_adh = trace_scores.adherence_score()
        binary_pred = 1 if comp_adh >= 0.5 else 0
        binary_gt = 1 if gt_adh >= 0.5 else 0

        return ExampleResult(
            example_idx=idx,
            dataset_id=dataset_id,
            question=question,
            ground_truth_answer=ground_truth_answer,
            generated_response=generated_response,
            retrieved_chunks=retrieved_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
            computed_relevance=trace_scores.context_relevance,
            computed_utilization=trace_scores.context_utilization,
            computed_completeness=trace_scores.completeness,
            computed_adherence=comp_adh,
            gt_relevance=gt_rel,
            gt_utilization=gt_util,
            gt_completeness=gt_comp,
            gt_adherence=gt_adh,
            binary_pred=binary_pred,
            binary_gt=binary_gt,
            is_valid=True
        )

    def evaluate_config(self, config: Tuple) -> Optional[ConfigResult]:
        """Evaluate a single configuration with 100 examples. Returns None if rate limited."""
        config_type, chunking, method, llm_model = config
        config_id = self._get_config_id(config)

        print(f"\n{'='*70}")
        print(f"📋 {config_type}/{chunking}/{method}/{llm_model.split('/')[-1]}")
        print(f"{'='*70}")

        checkpoint = self._load_checkpoint(config_id)
        completed = set(checkpoint["completed_examples"])
        example_results = {r["example_idx"]: r for r in checkpoint["example_results"]}
        failed_indices = set()

        print(f"  Checkpoint: {len(completed)}/{NUM_EXAMPLES} examples completed")

        try:
            for idx, sample in enumerate(self.samples):
                if idx in completed:
                    prev_result = example_results.get(idx, {})
                    if prev_result.get("is_valid", True):
                        trace = np.mean([
                            prev_result.get("computed_relevance", 0),
                            prev_result.get("computed_utilization", 0),
                            prev_result.get("computed_completeness", 0),
                            prev_result.get("computed_adherence", 0)
                        ])
                        if trace >= MIN_VALID_TRACE:
                            continue
                        elif RETRY_ON_ZERO_TRACE:
                            print(f"  Example {idx+1}: Previous TRACe={trace:.3f}, will retry...")
                            failed_indices.add(idx)
                            continue
                    else:
                        failed_indices.add(idx)
                        continue

                success = self._evaluate_with_retry(idx, sample, config, example_results)
                completed.add(idx)

                if not success:
                    failed_indices.add(idx)

                if len(completed) % CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(config_id, {
                        "completed_examples": list(completed),
                        "example_results": list(example_results.values())
                    })
                    print(f"  💾 Checkpoint saved: {len(completed)}/{NUM_EXAMPLES}")

            if failed_indices:
                print(f"\n  🔄 RETRY PASS: {len(failed_indices)} failed examples")

                for retry_round in range(MAX_EXAMPLE_RETRIES):
                    if not failed_indices:
                        break

                    print(f"  Retry round {retry_round + 1}/{MAX_EXAMPLE_RETRIES}...")
                    still_failed = set()

                    for idx in list(failed_indices):
                        sample = self.samples[idx]
                        time.sleep(RETRY_DELAY)

                        success = self._evaluate_with_retry(idx, sample, config, example_results)

                        if not success:
                            still_failed.add(idx)
                        else:
                            print(f"    ✓ Example {idx+1} succeeded on retry")

                    failed_indices = still_failed

                    self._save_checkpoint(config_id, {
                        "completed_examples": list(completed),
                        "example_results": list(example_results.values())
                    })

                if failed_indices:
                    print(f"  ⚠ {len(failed_indices)} examples still failed after retries: {sorted(failed_indices)}")

        except RateLimitExit as e:
            print(f"\n  ⚠️ RATE LIMIT HIT: {str(e)[:80]}")
            print(f"  💾 Saving checkpoint before exit...")
            self._save_checkpoint(config_id, {
                "completed_examples": list(completed),
                "example_results": list(example_results.values())
            })
            print(f"  ✓ Checkpoint saved: {len(completed)}/{NUM_EXAMPLES} examples completed")
            print(f"\n{'='*70}")
            print("⏸️  EVALUATION PAUSED - Rate limit reached")
            print("="*70)
            print(f"\nPlease wait 1-2 minutes and re-run:")
            print(f"  python -m src.run_final_evaluation --config {TOP_10_CONFIGS.index(config) + 1}")
            print(f"\nYour progress has been saved. Evaluation will resume from example {len(completed) + 1}.")
            return None

        self._save_checkpoint(config_id, {
            "completed_examples": list(completed),
            "example_results": list(example_results.values())
        })

        return self._aggregate_results(config, list(example_results.values()))

    def _evaluate_with_retry(self, idx: int, sample: Dict, config: Tuple,
                              example_results: Dict) -> bool:
        """Evaluate a single example with retry logic. Returns True if successful."""
        print(f"  Example {idx+1}/{NUM_EXAMPLES}...", end=" ", flush=True)

        for attempt in range(MAX_EXAMPLE_RETRIES + 1):
            try:
                result = self.evaluate_example(idx, sample, config)

                trace = np.mean([
                    result.computed_relevance,
                    result.computed_utilization,
                    result.computed_completeness,
                    result.computed_adherence
                ])

                if trace < MIN_VALID_TRACE and RETRY_ON_ZERO_TRACE and attempt < MAX_EXAMPLE_RETRIES:
                    print(f"TRACe={trace:.3f} (too low), retrying...", end=" ", flush=True)
                    time.sleep(RETRY_DELAY)
                    continue

                example_results[idx] = asdict(result)
                print(f"✓ TRACe={trace:.3f}")

                time.sleep(DELAY_BETWEEN_CALLS)
                return True

            except RateLimitExit:
                raise

            except Exception as e:
                error_msg = str(e)[:50]
                if attempt < MAX_EXAMPLE_RETRIES:
                    print(f"✗ {error_msg}, retrying...", end=" ", flush=True)
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"✗ {error_msg} (final)")
                    # Store failed result with all fields
                    example_results[idx] = {
                        "example_idx": idx,
                        "dataset_id": self.sample_ids[idx],
                        "question": sample.get('question', ''),
                        "ground_truth_answer": sample.get('response', sample.get('answer', '')),
                        "generated_response": "",
                        "retrieved_chunks": [],
                        "retrieved_chunk_ids": [],
                        "is_valid": False,
                        "error": str(e)[:200],
                        "computed_relevance": 0.0,
                        "computed_utilization": 0.0,
                        "computed_completeness": 0.0,
                        "computed_adherence": 0.0,
                        "gt_relevance": 0.0,
                        "gt_utilization": 0.0,
                        "gt_completeness": 0.0,
                        "gt_adherence": 0.0,
                        "binary_pred": 0,
                        "binary_gt": 0
                    }
                    return False

        return False

    def _aggregate_results(self, config: Tuple, example_results: List[Dict]) -> ConfigResult:
        """Aggregate example results into config metrics."""
        config_type, chunking, method, llm_model = config

        valid_results = [r for r in example_results if r.get("is_valid", False)]

        if not valid_results:
            return ConfigResult(
                config_type=config_type,
                chunking_strategy=chunking,
                retrieval_method=method,
                llm_model=llm_model,
                num_examples=len(example_results),
                num_valid=0,
                num_failed=len(example_results)
            )

        comp_rel = [r["computed_relevance"] for r in valid_results]
        comp_util = [r["computed_utilization"] for r in valid_results]
        comp_comp = [r["computed_completeness"] for r in valid_results]
        comp_adh = [r["computed_adherence"] for r in valid_results]

        gt_rel = [r["gt_relevance"] for r in valid_results]
        gt_util = [r["gt_utilization"] for r in valid_results]
        gt_comp = [r["gt_completeness"] for r in valid_results]
        gt_adh = [r["gt_adherence"] for r in valid_results]

        binary_pred = [r["binary_pred"] for r in valid_results]
        binary_gt = [r["binary_gt"] for r in valid_results]

        def rmse(computed, gt):
            return float(np.sqrt(np.mean((np.array(computed) - np.array(gt)) ** 2)))

        try:
            auroc = roc_auc_score(binary_gt, comp_adh) if len(set(binary_gt)) > 1 else 0.5
        except:
            auroc = 0.5

        try:
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            prec = precision_score(binary_gt, binary_pred, zero_division=0)
            rec = recall_score(binary_gt, binary_pred, zero_division=0)
        except:
            f1, prec, rec = 0.0, 0.0, 0.0

        return ConfigResult(
            config_type=config_type,
            chunking_strategy=chunking,
            retrieval_method=method,
            llm_model=llm_model,
            avg_relevance=float(np.mean(comp_rel)),
            avg_utilization=float(np.mean(comp_util)),
            avg_completeness=float(np.mean(comp_comp)),
            avg_adherence=float(np.mean(comp_adh)),
            gt_avg_relevance=float(np.mean(gt_rel)),
            gt_avg_utilization=float(np.mean(gt_util)),
            gt_avg_completeness=float(np.mean(gt_comp)),
            gt_avg_adherence=float(np.mean(gt_adh)),
            rmse_relevance=rmse(comp_rel, gt_rel),
            rmse_utilization=rmse(comp_util, gt_util),
            rmse_completeness=rmse(comp_comp, gt_comp),
            auroc=auroc,
            f1=f1,
            precision=prec,
            recall=rec,
            num_examples=len(example_results),
            num_valid=len(valid_results),
            num_failed=len(example_results) - len(valid_results),
            example_results=example_results
        )

    def run_evaluation(self):
        """Run evaluation for all top 10 configurations."""
        print("\n" + "="*70)
        print("🚀 FINAL EVALUATION - TOP 10 CONFIGS × 100 EXAMPLES")
        print("="*70)
        print(f"  Configurations: {len(TOP_10_CONFIGS)}")
        print(f"  Examples each:  {NUM_EXAMPLES}")
        print(f"  Total evals:    {len(TOP_10_CONFIGS) * NUM_EXAMPLES}")
        print(f"  Checkpoint:     Every {CHECKPOINT_INTERVAL} examples")

        results = []

        for i, config in enumerate(TOP_10_CONFIGS):
            print(f"\n[{i+1}/{len(TOP_10_CONFIGS)}]", end="")
            result = self.evaluate_config(config)

            if result is None:
                return results if results else None

            results.append(result)
            self._save_results(results)

        return results

    def _save_results(self, results: List[ConfigResult]):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"final_results_{timestamp}.json")
        latest_file = os.path.join(RESULTS_DIR, "final_results_latest.json")

        data = {
            "timestamp": timestamp,
            "num_configs": len(results),
            "examples_per_config": NUM_EXAMPLES,
            "results": [asdict(r) for r in results]
        }

        for filepath in [results_file, latest_file]:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

    def print_results(self, results: List[ConfigResult]):
        """Print results summary."""
        print("\n" + "="*120)
        print("📊 FINAL RESULTS - TOP 10 CONFIGURATIONS (100 examples each)")
        print("="*120)

        # Sort by avg_relevance as primary metric (since we removed avg_trace)
        sorted_results = sorted(results, key=lambda x: x.avg_relevance, reverse=True)

        print(f"\n{'Rank':<5} {'Retriever':<10} {'Method':<15} {'LLM':<15} "
              f"{'Avg Rel':<10} {'Avg Util':<10} {'Avg Comp':<10} {'Avg Adh':<10}")
        print("-"*95)

        for i, r in enumerate(sorted_results, 1):
            llm_short = r.llm_model.split('/')[-1][:13]
            print(f"{i:<5} {r.config_type:<10} {r.retrieval_method:<15} {llm_short:<15} "
                  f"{r.avg_relevance:<10.4f} {r.avg_utilization:<10.4f} {r.avg_completeness:<10.4f} "
                  f"{r.avg_adherence:<10.4f}")

        print("\n" + "-"*120)
        print(f"{'Rank':<5} {'Retriever':<10} {'Method':<15} {'LLM':<15} "
              f"{'RMSE_Rel':<10} {'RMSE_Util':<10} {'RMSE_Comp':<10} "
              f"{'AUROC':<8} {'F1':<8} {'Prec':<8} {'Recall':<8}")
        print("-"*120)

        for i, r in enumerate(sorted_results, 1):
            llm_short = r.llm_model.split('/')[-1][:13]
            print(f"{i:<5} {r.config_type:<10} {r.retrieval_method:<15} {llm_short:<15} "
                  f"{r.rmse_relevance:<10.4f} {r.rmse_utilization:<10.4f} {r.rmse_completeness:<10.4f} "
                  f"{r.auroc:<8.4f} {r.f1:<8.4f} {r.precision:<8.4f} {r.recall:<8.4f}")

        print("-"*120)

        print("\n🏆 BEST CONFIGURATIONS:")
        best_rel = max(results, key=lambda x: x.avg_relevance)
        best_util = max(results, key=lambda x: x.avg_utilization)
        best_comp = max(results, key=lambda x: x.avg_completeness)
        best_adh = max(results, key=lambda x: x.avg_adherence)

        print(f"  Highest Relevance: {best_rel.config_type}/{best_rel.retrieval_method} + {best_rel.llm_model.split('/')[-1]} = {best_rel.avg_relevance:.4f}")
        print(f"  Highest Utilization: {best_util.config_type}/{best_util.retrieval_method} + {best_util.llm_model.split('/')[-1]} = {best_util.avg_utilization:.4f}")
        print(f"  Highest Completeness: {best_comp.config_type}/{best_comp.retrieval_method} + {best_comp.llm_model.split('/')[-1]} = {best_comp.avg_completeness:.4f}")
        print(f"  Highest Adherence: {best_adh.config_type}/{best_adh.retrieval_method} + {best_adh.llm_model.split('/')[-1]} = {best_adh.avg_adherence:.4f}")

        best_auroc = max(results, key=lambda x: x.auroc)
        print(f"  Highest AUROC: {best_auroc.config_type}/{best_auroc.retrieval_method} + {best_auroc.llm_model.split('/')[-1]} = {best_auroc.auroc:.4f}")

        best_rmse_r = min(results, key=lambda x: x.rmse_relevance)
        best_rmse_u = min(results, key=lambda x: x.rmse_utilization)
        best_rmse_c = min(results, key=lambda x: x.rmse_completeness)

        print(f"\n📉 LOWEST RMSE (closest to ground truth):")
        print(f"  RMSE_Relevance: {best_rmse_r.config_type}/{best_rmse_r.retrieval_method} + {best_rmse_r.llm_model.split('/')[-1]} = {best_rmse_r.rmse_relevance:.4f}")
        print(f"  RMSE_Utilization: {best_rmse_u.config_type}/{best_rmse_u.retrieval_method} + {best_rmse_u.llm_model.split('/')[-1]} = {best_rmse_u.rmse_utilization:.4f}")
        print(f"  RMSE_Completeness: {best_rmse_c.config_type}/{best_rmse_c.retrieval_method} + {best_rmse_c.llm_model.split('/')[-1]} = {best_rmse_c.rmse_completeness:.4f}")

        print(f"\n📈 EVALUATION STATISTICS:")
        total_valid = sum(r.num_valid for r in results)
        total_failed = sum(r.num_failed for r in results)
        print(f"  Total examples evaluated: {total_valid + total_failed}")
        print(f"  Successful: {total_valid} ({100*total_valid/(total_valid+total_failed):.1f}%)")
        print(f"  Failed: {total_failed} ({100*total_failed/(total_valid+total_failed):.1f}%)")


def show_progress():
    """Show evaluation progress."""
    print("\n📈 EVALUATION PROGRESS")
    print("="*70)

    total_examples = len(TOP_10_CONFIGS) * NUM_EXAMPLES
    completed = 0
    failed_total = 0

    for config in TOP_10_CONFIGS:
        config_id = f"{config[0]}_{config[1]}_{config[2]}_{config[3].split('/')[-1]}"
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{config_id}.json")

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                done = len(data.get("completed_examples", []))
                results = data.get("example_results", [])

                failed = sum(1 for r in results if not r.get("is_valid", True))
                zero_trace = sum(1 for r in results if r.get("is_valid", True) and
                               np.mean([r.get("computed_relevance", 0),
                                       r.get("computed_utilization", 0),
                                       r.get("computed_completeness", 0),
                                       r.get("computed_adherence", 0)]) < MIN_VALID_TRACE)

                completed += done
                failed_total += failed + zero_trace

                if done >= NUM_EXAMPLES:
                    status = f"✓ Complete ({failed} failed, {zero_trace} zero)"
                else:
                    status = f"{done}/{NUM_EXAMPLES}"
        else:
            status = "Not started"

        print(f"  {config[0]:<8} {config[2]:<15} {config[3].split('/')[-1]:<20} {status}")

    print("-"*70)
    print(f"  Total: {completed}/{total_examples} ({100*completed/total_examples:.1f}%)")
    if failed_total > 0:
        print(f"  ⚠ Failed/Zero TRACe: {failed_total} (use --retry to retry)")


def retry_failed():
    """Retry all failed examples across all configurations."""
    print("\n🔄 RETRYING FAILED EXAMPLES")
    print("="*70)

    evaluator = FinalEvaluator()
    evaluator.load_data()
    evaluator.initialize_judge()
    evaluator.initialize_api_key()

    total_retried = 0
    total_fixed = 0

    for config in TOP_10_CONFIGS:
        config_id = f"{config[0]}_{config[1]}_{config[2]}_{config[3].split('/')[-1]}"
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{config_id}.json")

        if not os.path.exists(checkpoint_file):
            continue

        with open(checkpoint_file, 'r') as f:
            data = json.load(f)

        example_results = {r["example_idx"]: r for r in data.get("example_results", [])}

        failed_indices = []
        for idx, result in example_results.items():
            if not result.get("is_valid", True):
                failed_indices.append(idx)
            else:
                trace = np.mean([
                    result.get("computed_relevance", 0),
                    result.get("computed_utilization", 0),
                    result.get("computed_completeness", 0),
                    result.get("computed_adherence", 0)
                ])
                if trace < MIN_VALID_TRACE:
                    failed_indices.append(idx)

        if not failed_indices:
            continue

        print(f"\n{config[0]}/{config[2]}/{config[3].split('/')[-1]}: {len(failed_indices)} to retry")

        try:
            for idx in failed_indices:
                sample = evaluator.samples[idx]
                success = evaluator._evaluate_with_retry(idx, sample, config, example_results)
                total_retried += 1
                if success:
                    total_fixed += 1

            evaluator._save_checkpoint(config_id, {
                "completed_examples": data.get("completed_examples", []),
                "example_results": list(example_results.values())
            })

        except RateLimitExit as e:
            print(f"\n  ⚠️ Rate limit hit: {str(e)[:50]}")
            print(f"  💾 Saving progress and exiting...")
            evaluator._save_checkpoint(config_id, {
                "completed_examples": data.get("completed_examples", []),
                "example_results": list(example_results.values())
            })
            print(f"\n  Please wait and re-run: python -m src.run_final_evaluation --retry")
            break

    print(f"\n{'='*70}")
    print(f"Retried: {total_retried}, Fixed: {total_fixed}")
    print("="*70)


def show_results():
    """Show current results."""
    latest_file = os.path.join(RESULTS_DIR, "final_results_latest.json")

    if not os.path.exists(latest_file):
        print("❌ No results found. Run evaluation first.")
        return

    with open(latest_file, 'r') as f:
        data = json.load(f)

    results = []
    # Fields that have been removed from ConfigResult
    removed_fields = {'avg_trace', 'rmse_adherence'}

    for r in data["results"]:
        r_copy = {k: v for k, v in r.items() if k != 'example_results' and k not in removed_fields}
        r_copy['example_results'] = []
        results.append(ConfigResult(**r_copy))

    evaluator = FinalEvaluator()
    evaluator.print_results(results)


def list_configs():
    """List all configurations with their indices."""
    print("\n📋 TOP 10 CONFIGURATIONS")
    print("="*70)
    print(f"{'#':<4} {'Type':<8} {'Chunking':<10} {'Method':<15} {'LLM':<25}")
    print("-"*70)

    for i, config in enumerate(TOP_10_CONFIGS, 1):
        config_type, chunking, method, llm = config
        llm_short = llm.split('/')[-1]
        print(f"{i:<4} {config_type:<8} {chunking:<10} {method:<15} {llm_short:<25}")

    print("-"*70)
    print("\nUsage: python -m src.run_final_evaluation --config 1")
    print("       python -m src.run_final_evaluation --config 1,2,3")
    print("       python -m src.run_final_evaluation --config 1-5")


def main():
    parser = argparse.ArgumentParser(description="Final RAG Evaluation")
    parser.add_argument("--run", action="store_true", help="Run all configurations")
    parser.add_argument("--config", type=str, help="Run specific config(s): 1 or 1,2,3 or 1-5")
    parser.add_argument("--list", action="store_true", help="List all configurations")
    parser.add_argument("--progress", action="store_true", help="Show progress")
    parser.add_argument("--results", action="store_true", help="Show results")
    parser.add_argument("--retry", action="store_true", help="Retry all failed examples")

    args = parser.parse_args()

    if args.list:
        list_configs()
    elif args.progress:
        show_progress()
    elif args.results:
        show_results()
    elif args.retry:
        retry_failed()
    elif args.config:
        indices = []
        for part in args.config.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))

        indices = [i for i in indices if 1 <= i <= len(TOP_10_CONFIGS)]
        if not indices:
            print("❌ Invalid config indices. Use --list to see available configs.")
            return

        configs_to_run = [TOP_10_CONFIGS[i-1] for i in indices]

        print(f"\n🎯 Running {len(configs_to_run)} configuration(s): {indices}")

        evaluator = FinalEvaluator()
        evaluator.load_data()
        evaluator.initialize_judge()
        evaluator.initialize_api_key()

        results = []
        for i, config in enumerate(configs_to_run):
            print(f"\n[{i+1}/{len(configs_to_run)}]", end="")
            result = evaluator.evaluate_config(config)

            if result is None:
                sys.exit(0)

            results.append(result)
            evaluator._save_results(results)

        evaluator.print_results(results)

    elif args.run:
        evaluator = FinalEvaluator()
        evaluator.load_data()
        evaluator.initialize_judge()
        evaluator.initialize_api_key()
        results = evaluator.run_evaluation()
        if results:
            evaluator.print_results(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()