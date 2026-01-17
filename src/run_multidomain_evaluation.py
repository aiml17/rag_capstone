"""
Multi-Domain RAG Pipeline Full Evaluation
Tests top 10 configurations per domain with 100 examples each.

Structure:
- evaluations/
  - finqa/
    - checkpoints/
    - results/
  - cuad/
  - delucionqa/
  - hotpotqa/
  - covidqa/

Usage:
    python run_multidomain_evaluation.py --list                    # List domains & configs
    python run_multidomain_evaluation.py --domain finqa --run      # Run specific domain
    python run_multidomain_evaluation.py --domain finqa --progress # Check progress
    python run_multidomain_evaluation.py --domain finqa --results  # Show results
    python run_multidomain_evaluation.py --runall                  # Run all domains
    python run_multidomain_evaluation.py --compare                 # Cross-domain comparison
"""

import os
import sys
import json
import time
import shutil
import argparse
import uuid
import glob
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
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

from src.multidomain_loader import MultiDomainLoader
from src.chunker import DocumentChunker
from src.evaluator import TRACeEvaluator
from src.retriever import UnifiedRetriever
from src.hybrid_retriever import HybridRetriever
from groq import Groq

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_EXAMPLES = 100  # Full evaluation with 100 examples per config
SAMPLE_POOL_MULTIPLIER = 3  # Load 3x samples to have backups
MAX_CONTEXT_LENGTH = 20000  # Skip examples with context+question > this
CHECKPOINT_INTERVAL = 4
RANDOM_SEED = 42
TOP_K = 5
DELAY_BETWEEN_CALLS = 2.0

# Judge model (local Ollama)
JUDGE_MODEL = "qwen2.5:7b-instruct"
USE_LOCAL_JUDGE = True

# Retry settings
MIN_VALID_TRACE = 0.01

# Base directory for all evaluations
BASE_DIR = "evaluations"


class RateLimitExit(Exception):
    """Exception raised when rate limit is hit to trigger graceful exit."""
    pass


# ============================================================================
# TOP 10 CONFIGURATIONS PER DOMAIN (from preliminary evaluation)
# Format: (config_type, chunking, retrieval_method, llm_model)
# ============================================================================

DOMAIN_CONFIGS = {
    "finqa": [
        ("sparse", "semantic", "tfidf", "groq/compound"),
        ("hybrid", "semantic", "minilm", "openai/gpt-oss-120b"),
        ("sparse", "semantic", "bm25", "openai/gpt-oss-20b"),
        ("hybrid", "semantic", "e5-large", "groq/compound"),
        ("dense", "semantic", "e5-large", "qwen/qwen3-32b"),
        ("dense", "semantic", "mpnet", "groq/compound"),
        ("sparse", "semantic", "bm25", "groq/compound"),
        ("hybrid", "semantic", "finbert", "openai/gpt-oss-120b"),
        ("hybrid", "semantic", "mpnet", "groq/compound"),
        ("hybrid", "semantic", "gte-large", "openai/gpt-oss-20b"),
    ],

    "cuad": [
        ("dense", "semantic", "minilm", "openai/gpt-oss-120b"),
        ("dense", "semantic", "e5-large", "moonshotai/kimi-k2-instruct-0905"),
        ("hybrid", "semantic", "mpnet", "qwen/qwen3-32b"),
        ("dense", "semantic", "mpnet", "qwen/qwen3-32b"),
        ("dense", "semantic", "e5-large", "llama-3.3-70b-versatile"),
        ("dense", "semantic", "e5-large", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "mpnet", "llama-3.1-8b-instant"),
        ("dense", "semantic", "bge-large", "moonshotai/kimi-k2-instruct-0905"),
        ("hybrid", "semantic", "mpnet", "moonshotai/kimi-k2-instruct-0905"),
        ("hybrid", "semantic", "mpnet", "openai/gpt-oss-120b"),
    ],

    "delucionqa": [
        ("sparse", "semantic", "tfidf", "openai/gpt-oss-120b"),
        ("dense", "semantic", "gte-large", "openai/gpt-oss-120b"),
        ("sparse", "semantic", "tfidf", "groq/compound"),
        ("dense", "semantic", "gte-large", "qwen/qwen3-32b"),
        ("dense", "semantic", "e5-large", "qwen/qwen3-32b"),
        ("dense", "semantic", "minilm", "openai/gpt-oss-120b"),
        ("dense", "semantic", "bge-large", "qwen/qwen3-32b"),
        ("dense", "none", "minilm", "openai/gpt-oss-120b"),
        ("dense", "none", "bge-large", "openai/gpt-oss-120b"),
        ("dense", "semantic", "mpnet", "qwen/qwen3-32b"),
    ],

    "hotpotqa": [
        ("hybrid", "semantic", "bge-large", "qwen/qwen3-32b"),
        ("dense", "semantic", "bge-large", "qwen/qwen3-32b"),
        ("dense", "semantic", "bge-large", "openai/gpt-oss-120b"),
        ("sparse", "semantic", "bm25", "qwen/qwen3-32b"),
        ("dense", "semantic", "gte-large", "qwen/qwen3-32b"),
        ("sparse", "semantic", "tfidf", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "e5-large", "qwen/qwen3-32b"),
        ("dense", "sentence", "gte-large", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "gte-large", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "mpnet", "qwen/qwen3-32b"),
    ],

    "covidqa": [
        ("hybrid", "none", "mpnet", "qwen/qwen3-32b"),
        ("dense", "semantic", "biobert", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "sapbert", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "biobert", "qwen/qwen3-32b"),
        ("dense", "semantic", "mpnet", "openai/gpt-oss-120b"),
        ("hybrid", "semantic", "bge-base", "qwen/qwen3-32b"),
        ("hybrid", "semantic", "mpnet", "qwen/qwen3-32b"),
        ("dense", "none", "minilm", "qwen/qwen3-32b"),
        ("dense", "none", "sapbert", "qwen/qwen3-32b"),
        ("hybrid", "sentence", "biobert", "qwen/qwen3-32b"),
    ],
}

DOMAINS = list(DOMAIN_CONFIGS.keys())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_api_key() -> str:
    for i in range(1, 21):
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key:
            return key
    key = os.getenv('GROQ_API_KEY')
    if key:
        return key
    print("❌ No API key found!")
    sys.exit(1)


def cleanup_temp_folders():
    for pattern in ['temp_chroma_*', 'chroma_db_eval_*']:
        for folder in glob.glob(pattern):
            try:
                shutil.rmtree(folder)
            except:
                pass


def get_domain_dirs(domain: str) -> Tuple[str, str]:
    """Get checkpoint and results directories for a domain."""
    domain_dir = os.path.join(BASE_DIR, domain)
    checkpoint_dir = os.path.join(domain_dir, "checkpoints")
    results_dir = os.path.join(domain_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return checkpoint_dir, results_dir


def get_config_id(config: Tuple) -> str:
    """Get config ID string."""
    config_type, chunking, method, llm = config
    llm_short = llm.split("/")[-1]
    return f"{config_type}_{chunking}_{method}_{llm_short}"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExampleResult:
    example_idx: int
    dataset_id: str
    question: str
    ground_truth_answer: str
    generated_response: str
    retrieved_chunks: List[str]
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
    is_skipped: bool = False
    error: str = ""


@dataclass
class ConfigResult:
    config_type: str
    chunking_strategy: str
    retrieval_method: str
    llm_model: str
    domain: str

    avg_relevance: float = 0.0
    avg_utilization: float = 0.0
    avg_completeness: float = 0.0
    avg_adherence: float = 0.0
    avg_trace: float = 0.0

    gt_avg_relevance: float = 0.0
    gt_avg_utilization: float = 0.0
    gt_avg_completeness: float = 0.0
    gt_avg_adherence: float = 0.0

    rmse_relevance: float = 0.0
    rmse_utilization: float = 0.0
    rmse_completeness: float = 0.0

    auroc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    num_examples: int = 0
    num_valid: int = 0
    num_failed: int = 0
    num_skipped: int = 0
    example_results: List[Dict] = field(default_factory=list)


# ============================================================================
# DOMAIN EVALUATOR
# ============================================================================

class DomainEvaluator:
    """Evaluator for a single domain."""

    def __init__(self, domain: str):
        self.domain = domain
        self.domain_info = MultiDomainLoader.get_domain_info(domain)
        self.checkpoint_dir, self.results_dir = get_domain_dirs(domain)

        self.samples = []
        self.configs = DOMAIN_CONFIGS[domain]

        # Create mapping: config -> index (1-based for display)
        self.config_to_index = {config: idx for idx, config in enumerate(self.configs, start=1)}

        self.loader = MultiDomainLoader()
        self.chunker = DocumentChunker()
        self.trace_evaluator = None
        self.groq_client = None

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        cleanup_temp_folders()

    def load_data(self):
        """Load samples for this domain with backup pool."""
        print(f"\n📂 Loading {self.domain_info['name']} ({self.domain})...")
        all_samples = self.loader.load_domain(self.domain, split="test")

        # Shuffle and take samples with backups
        indices = list(range(len(all_samples)))
        random.shuffle(indices)

        sample_count = min(NUM_EXAMPLES * SAMPLE_POOL_MULTIPLIER, len(all_samples))
        self.samples = [all_samples[i] for i in indices[:sample_count]]

        print(f"  Loaded {len(self.samples)} examples as pool (from {len(all_samples)} total)")
        print(f"  Will use {NUM_EXAMPLES} per config, with {len(self.samples) - NUM_EXAMPLES} backups")

    def initialize_judge(self):
        print(f"\n🔧 Initializing judge model: {JUDGE_MODEL}")
        self.trace_evaluator = TRACeEvaluator(
            judge_model=JUDGE_MODEL,
            use_local=USE_LOCAL_JUDGE,
            verbose=False
        )

    def initialize_api(self):
        api_key = get_api_key()
        self.groq_client = Groq(api_key=api_key)
        print(f"✅ Groq API initialized")

    def _get_numbered_filename(self, config: Tuple) -> str:
        """Get checkpoint filename with number prefix."""
        idx = self.config_to_index.get(config, 0)
        base_id = get_config_id(config)
        return f"{idx:02d}_{base_id}"

    def _load_checkpoint(self, config: Tuple) -> Dict:
        """Load checkpoint for a config."""
        numbered_name = self._get_numbered_filename(config)
        filepath = os.path.join(self.checkpoint_dir, f"{numbered_name}.json")

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {"results": [], "skipped_samples": []}

    def _save_checkpoint(self, config: Tuple, data: Dict):
        """Save checkpoint for a config."""
        numbered_name = self._get_numbered_filename(config)
        filepath = os.path.join(self.checkpoint_dir, f"{numbered_name}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_dataset_id(self, sample: Dict, idx: int) -> str:
        for key in ['id', 'idx', 'example_id', 'qid', 'question_id', 'index']:
            if key in sample:
                return str(sample[key])
        return f"{self.domain}_{idx}"

    def _get_ground_truth_answer(self, sample: Dict) -> str:
        original = sample.get('_original', {})
        if isinstance(original, dict):
            for key in ['response', 'answer', 'ground_truth', 'expected_answer', 'gold_answer', 'reference']:
                if key in original and original[key]:
                    return str(original[key])
        for key in ['response', 'answer', 'ground_truth', 'expected_answer', 'gold_answer', 'reference']:
            if key in sample and sample[key]:
                return str(sample[key])
        return ""

    def _get_gt_scores(self, sample: Dict) -> Dict[str, float]:
        return {
            'relevance': float(sample.get('context_relevance', sample.get('relevance_score', 0.5))),
            'utilization': float(sample.get('context_utilization', sample.get('utilization_score', 0.5))),
            'completeness': float(sample.get('completeness_score', sample.get('completeness', 0.5))),
            'adherence': float(sample.get('adherence_score', 1.0 if sample.get('adherence', True) else 0.0)),
        }

    def _create_retriever(self, config: Tuple, chunks: List[str]):
        config_type, _, method, _ = config

        if config_type == "dense":
            temp_dir = f"temp_chroma_{uuid.uuid4().hex[:8]}"
            retriever = UnifiedRetriever(
                method=method,
                persist_directory=temp_dir,
                collection_prefix=f"eval_{self.domain}"
            )
            retriever.index_documents(chunks=chunks, clear_existing=True)
            return retriever

        elif config_type == "sparse":
            retriever = UnifiedRetriever(method=method)
            retriever.index_documents(chunks=chunks)
            return retriever

        elif config_type == "hybrid":
            retriever = HybridRetriever(
                dense_model=method,
                sparse_method='tfidf',
                alpha=0.5,
                verbose=False
            )
            retriever.index(chunks)
            return retriever

        raise ValueError(f"Unknown config type: {config_type}")

    def _retrieve(self, retriever, query: str, config_type: str) -> List[str]:
        try:
            results = retriever.retrieve(query, top_k=TOP_K)
            if not results:
                return []
            if hasattr(results[0], 'content'):
                return [r.content for r in results]
            elif isinstance(results[0], dict):
                return [r.get('content', r.get('text', r.get('chunk', str(r)))) for r in results]
            else:
                return [str(r) for r in results]
        except Exception as e:
            print(f"  ⚠ Retrieval error: {e}")
            return []

    def _generate_response(self, question: str, context: str, llm_model: str) -> Tuple[str, bool]:
        """Generate response. Returns (response, is_rate_limited)."""
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        try:
            time.sleep(DELAY_BETWEEN_CALLS)
            response = self.groq_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip(), False
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in str(e) or "too many requests" in error_str:
                print(f"\n  🛑 Rate limit hit!")
                return "", True
            else:
                print(f"\n  ⚠ Generation error: {e}")
                return "", False

    def evaluate_example(self, sample_idx: int, sample: Dict, config: Tuple, result_idx: int) -> ExampleResult:
        """Evaluate a single example."""
        config_type, chunking, method, llm = config
        dataset_id = self._get_dataset_id(sample, sample_idx)
        question = sample.get('question', '')
        ground_truth = self._get_ground_truth_answer(sample)
        documents = sample.get('documents', sample.get('context', []))
        gt_scores = self._get_gt_scores(sample)

        if isinstance(documents, str):
            documents = [documents]
        corpus = "\n\n".join(documents)

        # Chunking
        if chunking == "none":
            chunks = [corpus] if corpus else ["No content"]
        elif chunking == "sentence":
            chunk_results = self.chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]
        else:  # semantic
            chunk_results = self.chunker.semantic_chunking(corpus)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]

        if not chunks:
            chunks = [corpus] if corpus else ["No content"]

        cleanup_temp_folders()
        retriever = self._create_retriever(config, chunks)
        retrieved = self._retrieve(retriever, question, config_type)
        context = "\n\n".join(retrieved) if retrieved else corpus[:2000]
        cleanup_temp_folders()

        # Check if context + question exceeds limit BEFORE calling LLM
        total_length = len(context) + len(question)
        if total_length > MAX_CONTEXT_LENGTH:
            return ExampleResult(
                example_idx=result_idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response="",
                retrieved_chunks=retrieved,
                computed_relevance=0.0, computed_utilization=0.0,
                computed_completeness=0.0, computed_adherence=0.0,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=0, binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=False, is_skipped=True,
                error=f"Context too large ({total_length} > {MAX_CONTEXT_LENGTH})"
            )

        response, is_rate_limited = self._generate_response(question, context, llm)

        if is_rate_limited:
            raise RateLimitExit("Rate limit hit")

        if not response:
            return ExampleResult(
                example_idx=result_idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response="",
                retrieved_chunks=retrieved,
                computed_relevance=0.0, computed_utilization=0.0,
                computed_completeness=0.0, computed_adherence=0.0,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=0, binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=False, is_skipped=False, error="Empty response"
            )

        # Evaluate with TRACe
        try:
            result = self.trace_evaluator.evaluate_single(
                question=question, ground_truth=ground_truth,
                generated_response=response, retrieved_chunks=retrieved
            )
            scores = result.trace_scores
            comp_adh = scores.adherence_score()

            return ExampleResult(
                example_idx=result_idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response=response,
                retrieved_chunks=retrieved,
                computed_relevance=scores.context_relevance,
                computed_utilization=scores.context_utilization,
                computed_completeness=scores.completeness,
                computed_adherence=comp_adh,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=1 if comp_adh >= 0.5 else 0,
                binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=True, is_skipped=False, error=""
            )
        except Exception as e:
            return ExampleResult(
                example_idx=result_idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response=response,
                retrieved_chunks=retrieved,
                computed_relevance=0.0, computed_utilization=0.0,
                computed_completeness=0.0, computed_adherence=0.0,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=0, binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=False, is_skipped=False, error=str(e)
            )

    def evaluate_config(self, config: Tuple) -> Optional[ConfigResult]:
        """Evaluate a single configuration."""
        config_type, chunking, method, llm = config
        config_idx = self.config_to_index.get(config, 0)
        numbered_name = self._get_numbered_filename(config)
        start_time = datetime.now()
        start_perf = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"📋 [{config_idx:02d}] {config_type}/{chunking}/{method}/{llm.split('/')[-1]}")
        print(f"Start time: {start_time.strftime('%d-%m-%Y %H:%M:%S')}")
        print(f"{'=' * 60}")

        checkpoint = self._load_checkpoint(config)
        completed_count = len(checkpoint.get("results", []))
        results = checkpoint["results"]
        skipped_samples = set(checkpoint.get("skipped_samples", []))

        print(f"  Checkpoint: {completed_count}/{NUM_EXAMPLES} done, {len(skipped_samples)} skipped")

        # Continue from where we left off
        sample_idx = 0
        result_idx = completed_count

        try:
            while result_idx < NUM_EXAMPLES and sample_idx < len(self.samples):
                # Skip already processed samples
                if sample_idx in skipped_samples:
                    sample_idx += 1
                    continue

                sample = self.samples[sample_idx]
                print(f"  Example {result_idx + 1}/{NUM_EXAMPLES} (sample #{sample_idx})...", end=" ", flush=True)

                result = self.evaluate_example(sample_idx, sample, config, result_idx)

                # Handle skipped (too large)
                if result.is_skipped:
                    print(f"⚠ Too large - skipping")
                    skipped_samples.add(sample_idx)
                    self._save_checkpoint(config, {
                        "results": results,
                        "skipped_samples": list(skipped_samples)
                    })
                    sample_idx += 1
                    continue

                # Check for zero TRACe - skip and try next
                if result.is_valid:
                    avg = np.mean([result.computed_relevance, result.computed_utilization,
                                   result.computed_completeness, result.computed_adherence])
                    print(f"✔ TRACe={avg:.3f}")
                    if avg < MIN_VALID_TRACE:
                        sample_idx += 1
                        continue
                else:
                    print(f"✗ Failed ({result.error})")
                    sample_idx += 1
                    continue

                # Valid result
                results.append(asdict(result))
                result_idx += 1
                sample_idx += 1

                # Save checkpoint periodically
                if len(results) % CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(config, {
                        "results": results,
                        "skipped_samples": list(skipped_samples)
                    })
                    print(f"  💾 Checkpoint saved: {len(results)}/{NUM_EXAMPLES}")

        except RateLimitExit:
            self._save_checkpoint(config, {
                "results": results,
                "skipped_samples": list(skipped_samples)
            })
            print(f"\n  💾 Checkpoint saved: {len(results)}/{NUM_EXAMPLES} examples")
            raise

        # Final save
        self._save_checkpoint(config, {
            "results": results,
            "skipped_samples": list(skipped_samples)
        })

        if len(results) < NUM_EXAMPLES:
            print(f"  ⚠ Only got {len(results)}/{NUM_EXAMPLES} examples (ran out of samples)")

        # Aggregate results
        end_perf = time.perf_counter()
        end_time = datetime.now()
        duration_seconds = int(end_perf - start_perf)
        minutes, seconds = divmod(duration_seconds, 60)

        config_result = self._aggregate(config, results, len(skipped_samples))

        print(f"\n{'=' * 60}")
        print(f"  📊 Results: TRACe={config_result.avg_trace:.3f} "
              f"(R={config_result.avg_relevance:.2f}, U={config_result.avg_utilization:.2f}, "
              f"C={config_result.avg_completeness:.2f}, A={config_result.avg_adherence:.2f})")
        print(f"  ✅ Valid: {config_result.num_valid}/{len(results)}, Skipped: {len(skipped_samples)}")
        print(f"End time: {end_time.strftime('%d-%m-%Y %H:%M:%S')}")
        print(f"Time taken: {minutes:02d}:{seconds:02d}")
        print(f"{'=' * 60}")

        return config_result

    def _aggregate(self, config: Tuple, results: List[Dict], num_skipped: int) -> ConfigResult:
        """Aggregate results for a configuration."""
        valid = [r for r in results if r.get("is_valid", False)]

        if not valid:
            return ConfigResult(
                config_type=config[0], chunking_strategy=config[1],
                retrieval_method=config[2], llm_model=config[3],
                domain=self.domain, num_examples=len(results),
                num_valid=0, num_failed=len(results), num_skipped=num_skipped
            )

        def avg(key):
            return float(np.mean([r[key] for r in valid]))

        def rmse(c, g):
            return float(np.sqrt(np.mean((np.array([r[c] for r in valid]) -
                                          np.array([r[g] for r in valid])) ** 2)))

        avg_rel = avg("computed_relevance")
        avg_util = avg("computed_utilization")
        avg_comp = avg("computed_completeness")
        avg_adh = avg("computed_adherence")
        avg_trace = float(np.mean([avg_rel, avg_util, avg_comp, avg_adh]))

        binary_gt = [r["binary_gt"] for r in valid]
        binary_pred = [r["binary_pred"] for r in valid]
        comp_adh = [r["computed_adherence"] for r in valid]

        try:
            auroc = roc_auc_score(binary_gt, comp_adh) if len(set(binary_gt)) > 1 else 0.5
        except:
            auroc = 0.5

        try:
            f1 = f1_score(binary_gt, binary_pred, zero_division=0)
            prec = precision_score(binary_gt, binary_pred, zero_division=0)
            rec = recall_score(binary_gt, binary_pred, zero_division=0)
        except:
            f1, prec, rec = 0, 0, 0

        return ConfigResult(
            config_type=config[0], chunking_strategy=config[1],
            retrieval_method=config[2], llm_model=config[3],
            domain=self.domain,
            avg_relevance=avg_rel, avg_utilization=avg_util,
            avg_completeness=avg_comp, avg_adherence=avg_adh, avg_trace=avg_trace,
            gt_avg_relevance=avg("gt_relevance"), gt_avg_utilization=avg("gt_utilization"),
            gt_avg_completeness=avg("gt_completeness"), gt_avg_adherence=avg("gt_adherence"),
            rmse_relevance=rmse("computed_relevance", "gt_relevance"),
            rmse_utilization=rmse("computed_utilization", "gt_utilization"),
            rmse_completeness=rmse("computed_completeness", "gt_completeness"),
            auroc=auroc, f1=f1, precision=prec, recall=rec,
            num_examples=len(results), num_valid=len(valid),
            num_failed=len(results) - len(valid), num_skipped=num_skipped,
            example_results=results
        )

    def get_completed_configs(self) -> set:
        """Get set of completed config IDs."""
        completed = set()
        for f in glob.glob(os.path.join(self.checkpoint_dir, "*.json")):
            with open(f, 'r') as file:
                data = json.load(file)
                if len(data.get("results", [])) >= NUM_EXAMPLES:
                    filename = os.path.basename(f).replace(".json", "")
                    if filename[:2].isdigit() and filename[2] == '_':
                        config_id = filename[3:]
                    else:
                        config_id = filename
                    completed.add(config_id)
        return completed

    def run(self) -> List[ConfigResult]:
        """Run evaluation for all configs in this domain."""
        print(f"\n{'=' * 70}")
        print(f"🚀 FULL EVALUATION - {self.domain_info['name'].upper()} ({self.domain})")
        print(f"{'=' * 70}")
        print(f"  Configurations: {len(self.configs)}")
        print(f"  Examples each:  {NUM_EXAMPLES}")
        print(f"  Sample pool:    {NUM_EXAMPLES * SAMPLE_POOL_MULTIPLIER} (3x for backups)")
        print(f"  Max context:    {MAX_CONTEXT_LENGTH} chars")
        print(f"  Total evals:    {len(self.configs) * NUM_EXAMPLES}")

        self.load_data()
        self.initialize_judge()
        self.initialize_api()

        completed_ids = self.get_completed_configs()
        remaining = [c for c in self.configs if get_config_id(c) not in completed_ids]

        print(f"  Completed:      {len(completed_ids)}")
        print(f"  Remaining:      {len(remaining)}")
        print("=" * 70)

        if not remaining:
            print("\n✅ All configurations completed!")
            return self._load_all_results()

        results = []
        try:
            for i, config in enumerate(remaining):
                config_idx = self.config_to_index.get(config, 0)
                print(f"\n[{i + 1}/{len(remaining)}] Config #{config_idx:02d}")
                result = self.evaluate_config(config)
                if result:
                    results.append(result)
                self._save_results()

        except RateLimitExit:
            print(f"\n\n🛑 RATE LIMIT - EXITING")
            self._save_results()
            print(f"💾 All progress saved.")
            print(f"\n📋 To resume, run the same command again:")
            print(f"   python run_multidomain_evaluation.py --domain {self.domain} --run")
            return results

        except KeyboardInterrupt:
            print(f"\n\n⚠️ INTERRUPTED BY USER")
            self._save_results()
            print(f"\n📋 To resume, run the same command again.")
            return results

        print(f"\n✅ Evaluation complete for {self.domain}!")
        self._save_results()
        return self._load_all_results()

    def _load_all_results(self) -> List[ConfigResult]:
        """Load all results from checkpoints."""
        results = []
        for config in self.configs:
            checkpoint = self._load_checkpoint(config)
            if len(checkpoint.get("results", [])) > 0:
                num_skipped = len(checkpoint.get("skipped_samples", []))
                result = self._aggregate(config, checkpoint["results"], num_skipped)
                results.append(result)
        return results

    def _save_results(self):
        """Save aggregated results."""
        results = self._load_all_results()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            "domain": self.domain,
            "domain_name": self.domain_info["name"],
            "timestamp": timestamp,
            "num_configs": len(self.configs),
            "examples_per_config": NUM_EXAMPLES,
            "completed_configs": len(results),
            "results": [asdict(r) for r in results]
        }

        # Save timestamped
        filepath = os.path.join(self.results_dir, f"results_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        # Save latest
        latest = os.path.join(self.results_dir, "results_latest.json")
        with open(latest, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def print_results(self, results: List[ConfigResult]):
        """Print results table."""
        if not results:
            print("No results to display.")
            return

        print(f"\n{'=' * 130}")
        print(f"📊 {self.domain_info['name']} Results ({len(results)} configs, {NUM_EXAMPLES} examples each)")
        print(f"{'=' * 130}")

        sorted_results = sorted(results, key=lambda x: x.avg_trace, reverse=True)

        # Table 1: TRACe Scores
        print(f"\n📈 TRACe Scores (higher is better):")
        print(f"{'#':<3} {'Type':<7} {'Chunk':<10} {'Method':<12} {'LLM':<15} "
              f"{'Rel':<7} {'Util':<7} {'Comp':<7} {'Adh':<7} {'Valid':<6}")
        print("-" * 105)

        for i, r in enumerate(sorted_results, 1):
            llm = r.llm_model.split('/')[-1][:13]
            print(f"{i:<3} {r.config_type:<7} {r.chunking_strategy:<10} {r.retrieval_method:<12} {llm:<15} "
                  f"{r.avg_relevance:<7.4f} {r.avg_utilization:<7.4f} "
                  f"{r.avg_completeness:<7.4f} {r.avg_adherence:<7.4f} {r.num_valid:<6}")

        # Table 2: RMSE Scores
        print(f"\n📉 RMSE Scores (lower is better):")
        print(f"{'#':<3} {'Type':<7} {'Chunk':<10} {'Method':<12} {'LLM':<15} "
              f"{'RMSE_R':<9} {'RMSE_U':<9} {'RMSE_C':<9}")
        print("-" * 80)

        for i, r in enumerate(sorted_results, 1):
            llm = r.llm_model.split('/')[-1][:13]
            print(f"{i:<3} {r.config_type:<7} {r.chunking_strategy:<10} {r.retrieval_method:<12} {llm:<15} "
                  f"{r.rmse_relevance:<9.4f} {r.rmse_utilization:<9.4f} {r.rmse_completeness:<9.4f}")

        # Table 3: Classification Metrics
        print(f"\n🎯 Adherence Classification Metrics:")
        print(f"{'#':<3} {'Type':<7} {'Chunk':<10} {'Method':<12} {'LLM':<15} "
              f"{'AUROC':<8} {'F1':<8} {'Prec':<8} {'Recall':<8}")
        print("-" * 85)

        for i, r in enumerate(sorted_results, 1):
            llm = r.llm_model.split('/')[-1][:13]
            print(f"{i:<3} {r.config_type:<7} {r.chunking_strategy:<10} {r.retrieval_method:<12} {llm:<15} "
                  f"{r.auroc:<8.4f} {r.f1:<8.4f} {r.precision:<8.4f} {r.recall:<8.4f}")

        # Best configs
        print(f"\n{'=' * 70}")
        print(f"🏆 BEST CONFIGURATIONS FOR {self.domain_info['name'].upper()}:")
        print(f"{'=' * 70}")

        best = sorted_results[0]
        print(f"  Best Overall (TRACe): {best.config_type}/{best.retrieval_method} + {best.llm_model.split('/')[-1]}")
        print(f"    TRACe={best.avg_trace:.4f} (R={best.avg_relevance:.4f}, U={best.avg_utilization:.4f}, "
              f"C={best.avg_completeness:.4f}, A={best.avg_adherence:.4f})")


# ============================================================================
# CROSS-DOMAIN COMPARISON
# ============================================================================

def compare_domains():
    """Compare results across all domains."""
    print("\n" + "=" * 130)
    print("📊 CROSS-DOMAIN COMPARISON")
    print("=" * 130)

    domain_results = {}

    for domain in DOMAINS:
        _, results_dir = get_domain_dirs(domain)
        latest = os.path.join(results_dir, "results_latest.json")

        if os.path.exists(latest):
            with open(latest, 'r') as f:
                data = json.load(f)
                domain_results[domain] = data

    if not domain_results:
        print("No results found. Run evaluations first.")
        return

    # Best per domain
    print(f"\n🏆 BEST CONFIGURATION PER DOMAIN (by TRACe):")
    print("-" * 140)
    print(f"{'Domain':<15} {'Best Config':<45} {'TRACe':<8} {'Rel':<7} {'Util':<7} {'Comp':<7} {'Adh':<7} {'AUROC':<8}")
    print("-" * 140)

    for domain, data in domain_results.items():
        results = data.get("results", [])
        if results:
            best = max(results, key=lambda x: x.get("avg_trace", 0))
            config_str = f"{best['config_type']}/{best['chunking_strategy']}/{best['retrieval_method']}/{best['llm_model'].split('/')[-1]}"
            domain_name = MultiDomainLoader.get_domain_info(domain)["name"]
            print(f"{domain_name:<15} {config_str:<45} "
                  f"{best.get('avg_trace', 0):<8.4f} {best.get('avg_relevance', 0):<7.4f} "
                  f"{best.get('avg_utilization', 0):<7.4f} {best.get('avg_completeness', 0):<7.4f} "
                  f"{best.get('avg_adherence', 0):<7.4f} {best.get('auroc', 0):<8.4f}")

    # Insights
    print(f"\n💡 DOMAIN-SPECIFIC INSIGHTS:")
    print("-" * 80)
    for domain, data in domain_results.items():
        results = data.get("results", [])
        if results:
            best = max(results, key=lambda x: x.get("avg_trace", 0))
            config_type = best.get("config_type", "")
            method = best.get("retrieval_method", "")
            domain_name = MultiDomainLoader.get_domain_info(domain)["name"]

            if config_type == "sparse":
                insight = f"Sparse ({method}) optimal - lexical matching effective"
            elif config_type == "hybrid":
                insight = f"Hybrid ({method}) optimal - benefits from semantic + lexical"
            else:
                insight = f"Dense ({method}) optimal - semantic understanding is key"

            print(f"  • {domain_name}: {insight}")


def show_progress(domain: str = None):
    """Show progress for domain(s)."""
    domains_to_check = [domain] if domain else DOMAINS

    print("\n📈 EVALUATION PROGRESS")
    print("=" * 70)

    for d in domains_to_check:
        if d not in DOMAINS:
            continue

        info = MultiDomainLoader.get_domain_info(d)
        checkpoint_dir, _ = get_domain_dirs(d)
        configs = DOMAIN_CONFIGS[d]

        print(f"\n{info['name']} ({d}):")

        total_done = 0
        for i, config in enumerate(configs, 1):
            config_id = get_config_id(config)
            # Check both numbered and legacy formats
            patterns = [
                os.path.join(checkpoint_dir, f"{i:02d}_{config_id}.json"),
                os.path.join(checkpoint_dir, f"{config_id}.json")
            ]

            found = False
            for filepath in patterns:
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        done = len(data.get("results", []))
                        total_done += done
                        status = f"{done}/{NUM_EXAMPLES}" + (" ✔" if done >= NUM_EXAMPLES else "")
                    found = True
                    break

            if not found:
                status = "Not started"

            config_short = f"{config[0][:3]}/{config[2][:10]}/{config[3].split('/')[-1][:10]}"
            print(f"  [{i:02d}] {config_short:<35} {status}")

        expected = len(configs) * NUM_EXAMPLES
        pct = 100 * total_done / expected if expected > 0 else 0
        print(f"  --- Total: {total_done}/{expected} ({pct:.1f}%)")


def show_results(domain: str):
    """Show results for a domain."""
    _, results_dir = get_domain_dirs(domain)
    latest = os.path.join(results_dir, "results_latest.json")

    if not os.path.exists(latest):
        print(f"❌ No results for {domain}. Run evaluation first.")
        return

    with open(latest, 'r') as f:
        data = json.load(f)

    evaluator = DomainEvaluator(domain)
    results = []

    for r in data.get("results", []):
        r_clean = {k: v for k, v in r.items() if k != 'example_results'}
        r_clean['example_results'] = []
        results.append(ConfigResult(**r_clean))

    evaluator.print_results(results)


def list_all():
    """List all domains and configs."""
    print("\n📋 DOMAINS:")
    for d in DOMAINS:
        info = MultiDomainLoader.get_domain_info(d)
        print(f"  {d:<12} - {info['name']}")

    print(f"\n📋 TOP 10 CONFIGS PER DOMAIN:")
    for d in DOMAINS:
        info = MultiDomainLoader.get_domain_info(d)
        print(f"\n  {info['name']} ({d}):")
        for i, c in enumerate(DOMAIN_CONFIGS[d], 1):
            print(f"    {i:<3} {c[0]:<8} {c[1]:<10} {c[2]:<12} {c[3].split('/')[-1]}")

    total_evals = len(DOMAINS) * 10 * NUM_EXAMPLES
    print(f"\n📊 TOTAL: {len(DOMAINS)} domains × 10 configs × {NUM_EXAMPLES} examples = {total_evals} evaluations")


def main():
    parser = argparse.ArgumentParser(description="Multi-Domain RAG Full Evaluation")
    parser.add_argument("--domain", type=str, choices=DOMAINS, help="Domain to evaluate")
    parser.add_argument("--run", action="store_true", help="Run evaluation")
    parser.add_argument("--runall", action="store_true", help="Run all domains")
    parser.add_argument("--progress", action="store_true", help="Show progress")
    parser.add_argument("--results", action="store_true", help="Show results")
    parser.add_argument("--compare", action="store_true", help="Cross-domain comparison")
    parser.add_argument("--list", action="store_true", help="List domains & configs")

    args = parser.parse_args()

    if args.list:
        list_all()

    elif args.compare:
        compare_domains()

    elif args.progress:
        show_progress(args.domain)

    elif args.domain:
        if args.results:
            show_results(args.domain)
        elif args.run:
            evaluator = DomainEvaluator(args.domain)
            results = evaluator.run()
            if results:
                evaluator.print_results(results)
        else:
            print(f"Specify --run or --results for domain {args.domain}")

    elif args.runall:
        for domain in DOMAINS:
            print(f"\n{'#' * 70}")
            print(f"# DOMAIN: {domain.upper()}")
            print(f"{'#' * 70}")
            evaluator = DomainEvaluator(domain)
            results = evaluator.run()
            if results:
                evaluator.print_results(results)

        # Final comparison
        compare_domains()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()