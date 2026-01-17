"""
Preliminary Evaluation Script v10
Tests ALL configurations with 10 examples each to find top 10 per domain.

Features:
- Domain-specific embeddings and LLM models
- Checkpointing: Saves progress every 5 examples per config
- NUMBERED checkpoint files (e.g., 001_dense_none_bge-large_compound.json)
- Rate limit handling: Immediate exit on rate limit (saves progress first)
- Resume: Automatically continues from last checkpoint on restart

Usage:
    python run_preliminary_evaluation.py --list                    # List all domains and config counts
    python run_preliminary_evaluation.py --domain cuad --run       # Run evaluation for a domain
    python run_preliminary_evaluation.py --domain cuad --progress  # Check progress (includes partial)
    python run_preliminary_evaluation.py --domain cuad --results   # View results sorted by TRACe
    python run_preliminary_evaluation.py --domain cuad --top10     # Get top 10 configs to copy
"""

import os
import sys
import json
import time
import shutil
import argparse
import glob
import random
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import roc_auc_score

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

NUM_EXAMPLES = 10  # Examples per configuration for preliminary run
CHECKPOINT_INTERVAL = 5
RANDOM_SEED = 42
TOP_K = 5
DELAY_BETWEEN_CALLS = 2.0

# Judge model (local Ollama) - Using 7b for faster evaluation
JUDGE_MODEL = "qwen2.5:7b-instruct"
USE_LOCAL_JUDGE = True

# Retry settings
MAX_RETRIES = 2  # Only for non-rate-limit errors
RETRY_DELAY = 3

# Base directory
BASE_DIR = "preliminary_evaluations"


class RateLimitExit(Exception):
    """Exception raised when rate limit is hit to trigger graceful exit."""
    pass


# ============================================================================
# DOMAIN-SPECIFIC CONFIGURATIONS
# ============================================================================

CHUNKING_STRATEGIES = ["none", "sentence", "semantic"]
SPARSE_METHODS = ["bm25", "tfidf"]

DOMAIN_EMBEDDINGS = {
    "hotpotqa": ["bge-large", "e5-large", "gte-large",  "minilm", "mpnet"],
    "covidqa": ["bioclinicalbert", "pubmedbert", "biomed-e5-large", "sapbert", "bge-base", "biobert", "minilm", "mpnet"],
    "cuad": ["legal-bert", "law-embedding-1", "e5-large", "bge-large",  "minilm", "mpnet"],
    "finqa": ["finbert", "bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "delucionqa": ["bge-large", "e5-large", "gte-large", "minilm", "mpnet"]
}

DOMAIN_LLM_MODELS = {
    "hotpotqa": ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905", "llama-3.1-8b-instant"],
    "covidqa": ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905", "llama-3.1-8b-instant"],
    "cuad": ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905", "llama-3.1-8b-instant"],
    "finqa": ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "groq/compound", "openai/gpt-oss-20b"],
    "delucionqa": ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905", "groq/compound"]
}

DOMAINS = ["finqa", "cuad", "delucionqa", "hotpotqa", "covidqa"]


def generate_configs_for_domain(domain: str) -> List[Tuple[str, str, str, str]]:
    """Generate all configurations for a domain (deterministic order)."""
    configs = []
    embeddings = DOMAIN_EMBEDDINGS.get(domain, ["minilm", "mpnet"])
    llm_models = DOMAIN_LLM_MODELS.get(domain, ["llama-3.3-70b-versatile"])

    for chunking in CHUNKING_STRATEGIES:
        for emb in embeddings:
            for llm in llm_models:
                configs.append(("dense", chunking, emb, llm))
        for method in SPARSE_METHODS:
            for llm in llm_models:
                configs.append(("sparse", chunking, method, llm))
        for emb in embeddings:
            for llm in llm_models:
                configs.append(("hybrid", chunking, emb, llm))
    return configs


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
    print("âŒ No API key found! Set GROQ_API_KEY in .env")
    sys.exit(1)


def cleanup_temp_folders():
    for pattern in ['temp_chroma_*', 'chroma_db_*']:
        for folder in glob.glob(pattern):
            try:
                shutil.rmtree(folder)
            except:
                pass


def get_config_id(config: Tuple) -> str:
    """Get base config ID without number prefix."""
    config_type, chunking, method, llm = config
    llm_short = llm.split("/")[-1]
    return f"{config_type}_{chunking}_{method}_{llm_short}"


def get_domain_dirs(domain: str) -> Tuple[str, str]:
    checkpoint_dir = os.path.join(BASE_DIR, domain, "checkpoints")
    results_dir = os.path.join(BASE_DIR, domain, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return checkpoint_dir, results_dir


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
    is_rate_limited: bool = False
    error: str = ""


@dataclass
class ConfigResult:
    config_type: str
    chunking: str
    method: str
    llm_model: str
    avg_relevance: float
    avg_utilization: float
    avg_completeness: float
    avg_adherence: float
    avg_trace: float
    num_valid: int
    num_failed: int


# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class PreliminaryEvaluator:
    def __init__(self, domain: str):
        self.domain = domain
        self.checkpoint_dir, self.results_dir = get_domain_dirs(domain)
        self.loader = MultiDomainLoader()
        self.chunker = DocumentChunker()
        self.trace_evaluator = None
        self.groq_client = None
        self.samples = []
        self.all_configs = generate_configs_for_domain(domain)

        # Create mapping: config -> index (1-based for display)
        self.config_to_index = {config: idx for idx, config in enumerate(self.all_configs, start=1)}

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def _get_numbered_filename(self, config: Tuple) -> str:
        """Get checkpoint filename with number prefix (e.g., 001_dense_none_bge-large_compound.json)"""
        idx = self.config_to_index.get(config, 0)
        base_id = get_config_id(config)
        return f"{idx:03d}_{base_id}"

    def _load_checkpoint(self, config: Tuple) -> Dict:
        """Load checkpoint, checking both numbered and legacy formats."""
        numbered_name = self._get_numbered_filename(config)
        numbered_path = os.path.join(self.checkpoint_dir, f"{numbered_name}.json")

        # Try numbered format first
        if os.path.exists(numbered_path):
            with open(numbered_path, 'r') as f:
                return json.load(f)

        # Fallback: check legacy format (without number)
        legacy_name = get_config_id(config)
        legacy_path = os.path.join(self.checkpoint_dir, f"{legacy_name}.json")
        if os.path.exists(legacy_path):
            # Migrate to numbered format
            with open(legacy_path, 'r') as f:
                data = json.load(f)
            # Save in new format and delete old
            with open(numbered_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.remove(legacy_path)
            print(f"  ðŸ“ Migrated checkpoint to numbered format: {numbered_name}")
            return data

        return {"completed": [], "results": []}

    def _save_checkpoint(self, config: Tuple, data: Dict):
        """Save checkpoint with numbered filename."""
        numbered_name = self._get_numbered_filename(config)
        path = os.path.join(self.checkpoint_dir, f"{numbered_name}.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _create_retriever(self, config: Tuple, chunks: List[str]):
        config_type, chunking, method, llm = config

        if config_type == "sparse":
            retriever = UnifiedRetriever(method=method)
            retriever.index_documents(chunks=chunks)
            return retriever
        elif config_type == "dense":
            temp_dir = f"./temp_chroma_{uuid.uuid4().hex[:8]}"
            retriever = UnifiedRetriever(
                method=method,
                persist_directory=temp_dir,
                collection_prefix=f"eval_{self.domain}"
            )
            retriever.index_documents(chunks=chunks, clear_existing=True)
            return retriever
        else:  # hybrid
            retriever = HybridRetriever(
                dense_model=method,
                sparse_method="tfidf",
                alpha=0.5,
                verbose=False
            )
            retriever.index(chunks)
            return retriever

    def _retrieve(self, retriever, question: str, config_type: str) -> List[str]:
        try:
            results = retriever.retrieve(question, top_k=TOP_K)
            if not results:
                return []
            if hasattr(results[0], 'content'):
                return [r.content for r in results]
            elif isinstance(results[0], dict):
                return [r.get('content', r.get('text', r.get('chunk', str(r)))) for r in results]
            else:
                return [str(r) for r in results]
        except Exception as e:
            print(f"  âš  Retrieval error: {e}")
            return []

    def _generate_response(self, question: str, context: str, llm_model: str) -> Tuple[str, bool]:
        prompt = f"""Answer the question based on the provided context.

Context:
{context[:4000]}

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
                print(f"\n  ðŸ›‘ Rate limit hit! {e}")
                return "", True
            else:
                print(f"\n  âš  Generation error: {e}")
                return "", False

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

    def load_data(self):
        print(f"\nðŸ“‚ Loading {self.domain} data...")
        data = self.loader.load_domain(self.domain, "test")
        indices = list(range(len(data)))
        random.shuffle(indices)
        self.samples = [data[i] for i in indices[:NUM_EXAMPLES]]
        print(f"  Loaded {len(self.samples)} examples (from {len(data)} total)")

    def initialize_judge(self):
        print(f"\nðŸ”§ Initializing judge model: {JUDGE_MODEL}")
        self.trace_evaluator = TRACeEvaluator(
            judge_model=JUDGE_MODEL,
            use_local=USE_LOCAL_JUDGE,
            verbose=False
        )

    def initialize_api(self):
        api_key = get_api_key()
        self.groq_client = Groq(api_key=api_key)
        print(f"âœ… Groq API initialized")

    def evaluate_example(self, idx: int, sample: Dict, config: Tuple) -> ExampleResult:
        config_type, chunking, method, llm = config
        dataset_id = self._get_dataset_id(sample, idx)
        question = sample.get('question', '')
        ground_truth = self._get_ground_truth_answer(sample)
        documents = sample.get('documents', sample.get('context', []))
        gt_scores = self._get_gt_scores(sample)

        if isinstance(documents, str):
            documents = [documents]
        corpus = "\n\n".join(documents)

        if chunking == "none":
            chunks = [corpus] if corpus else ["No content"]
        elif chunking == "sentence":
            chunk_results = self.chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]
        else:
            chunk_results = self.chunker.semantic_chunking(corpus)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]

        if not chunks:
            chunks = [corpus] if corpus else ["No content"]

        cleanup_temp_folders()
        retriever = self._create_retriever(config, chunks)
        retrieved = self._retrieve(retriever, question, config_type)
        context = "\n\n".join(retrieved) if retrieved else corpus[:2000]
        cleanup_temp_folders()

        response, is_rate_limited = self._generate_response(question, context, llm)

        if is_rate_limited:
            return ExampleResult(
                example_idx=idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response="",
                retrieved_chunks=retrieved,
                computed_relevance=0.0, computed_utilization=0.0,
                computed_completeness=0.0, computed_adherence=0.0,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=0, binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=False, is_rate_limited=True, error="Rate limit hit"
            )

        if not response:
            return ExampleResult(
                example_idx=idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response="",
                retrieved_chunks=retrieved,
                computed_relevance=0.0, computed_utilization=0.0,
                computed_completeness=0.0, computed_adherence=0.0,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=0, binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=False, is_rate_limited=False, error="Empty response"
            )

        try:
            result = self.trace_evaluator.evaluate_single(
                question=question, ground_truth=ground_truth,
                generated_response=response, retrieved_chunks=retrieved
            )
            scores = result.trace_scores
            comp_adh = scores.adherence_score()

            return ExampleResult(
                example_idx=idx, dataset_id=dataset_id, question=question,
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
                is_valid=True, is_rate_limited=False, error=""
            )
        except Exception as e:
            return ExampleResult(
                example_idx=idx, dataset_id=dataset_id, question=question,
                ground_truth_answer=ground_truth, generated_response=response,
                retrieved_chunks=retrieved,
                computed_relevance=0.0, computed_utilization=0.0,
                computed_completeness=0.0, computed_adherence=0.0,
                gt_relevance=gt_scores['relevance'], gt_utilization=gt_scores['utilization'],
                gt_completeness=gt_scores['completeness'], gt_adherence=gt_scores['adherence'],
                binary_pred=0, binary_gt=1 if gt_scores['adherence'] >= 0.5 else 0,
                is_valid=False, is_rate_limited=False, error=str(e)
            )

    def evaluate_config(self, config: Tuple) -> Optional[ConfigResult]:
        config_type, chunking, method, llm = config
        config_idx = self.config_to_index.get(config, 0)
        numbered_name = self._get_numbered_filename(config)
        start_time = datetime.now()
        start_perf = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"ðŸ“‹ [{config_idx:03d}] {config_type}/{chunking}/{method}/{llm.split('/')[-1]}")
        print(f"Start time: {start_time.strftime('%d-%m-%Y %H:%M:%S')}")
        print(f"{'='*60}")

        checkpoint = self._load_checkpoint(config)
        completed = set(checkpoint["completed"])
        results = checkpoint["results"]

        print(f"  Checkpoint: {len(completed)}/{NUM_EXAMPLES} done")

        for idx, sample in enumerate(self.samples):
            if idx in completed:
                continue

            print(f"  Example {idx + 1}/{NUM_EXAMPLES}...", end=" ", flush=True)

            result = self.evaluate_example(idx, sample, config)

            if result.is_rate_limited:
                self._save_checkpoint(config, {"completed": list(completed), "results": results})
                print(f"\n  ðŸ’¾ Checkpoint saved: {len(completed)}/{NUM_EXAMPLES} examples")
                raise RateLimitExit(f"Rate limit hit on {numbered_name}")

            results.append(asdict(result))
            completed.add(idx)

            if result.is_valid:
                avg = np.mean([result.computed_relevance, result.computed_utilization,
                              result.computed_completeness, result.computed_adherence])
                print(f"âœ” TRACe={avg:.3f}")
            else:
                print(f"âœ— Failed ({result.error})")

            if len(completed) % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(config, {"completed": list(completed), "results": results})
                print(f"  ðŸ’¾ Checkpoint saved")

        self._save_checkpoint(config, {"completed": list(completed), "results": results})

        valid = [r for r in results if r["is_valid"]]
        if not valid:
            print(f"  âš  No valid results")
            return None

        avg_rel = float(np.mean([r["computed_relevance"] for r in valid]))
        avg_util = float(np.mean([r["computed_utilization"] for r in valid]))
        avg_comp = float(np.mean([r["computed_completeness"] for r in valid]))
        avg_adh = float(np.mean([r["computed_adherence"] for r in valid]))
        avg_trace = float(np.mean([avg_rel, avg_util, avg_comp, avg_adh]))
        end_perf = time.perf_counter()
        end_time = datetime.now()
        duration_seconds = int(end_perf - start_perf)
        minutes, seconds = divmod(duration_seconds, 60)

        print(f"\n{'=' * 60}")
        print(f"\n  ðŸ“Š Results: TRACe={avg_trace:.3f} (R={avg_rel:.2f}, U={avg_util:.2f}, C={avg_comp:.2f}, A={avg_adh:.2f})")
        print(f"  âœ… Valid: {len(valid)}/{len(results)}")
        print(f"End time: {end_time.strftime('%d-%m-%Y %H:%M:%S')}")
        print(f"Time taken: {minutes:02d}:{seconds:02d}")
        print(f"\n{'=' * 60}")

        return ConfigResult(
            config_type=config_type, chunking=chunking, method=method, llm_model=llm,
            avg_relevance=avg_rel, avg_utilization=avg_util, avg_completeness=avg_comp,
            avg_adherence=avg_adh, avg_trace=avg_trace,
            num_valid=len(valid), num_failed=len(results) - len(valid)
        )

    def get_completed_configs(self) -> set:
        """Get set of completed config indices (numbered)."""
        completed = set()
        for f in glob.glob(os.path.join(self.checkpoint_dir, "*.json")):
            with open(f, 'r') as file:
                data = json.load(file)
                if len(data.get("completed", [])) >= NUM_EXAMPLES:
                    # Extract the base config ID (remove number prefix if present)
                    filename = os.path.basename(f).replace(".json", "")
                    # Handle both numbered (001_xxx) and legacy (xxx) formats
                    if filename[:3].isdigit() and filename[3] == '_':
                        config_id = filename[4:]  # Remove "XXX_" prefix
                    else:
                        config_id = filename
                    completed.add(config_id)
        return completed

    def run_evaluation(self):
        print("\n" + "=" * 70)
        print(f"ðŸš€ PRELIMINARY EVALUATION - {self.domain.upper()}")
        print("=" * 70)
        print(f"  Configurations: {len(self.all_configs)}")
        print(f"  Examples each:  {NUM_EXAMPLES}")
        print(f"  Total evals:    {len(self.all_configs) * NUM_EXAMPLES}")
        print(f"  Embeddings:     {len(DOMAIN_EMBEDDINGS[self.domain])}")
        print(f"  LLMs:           {len(DOMAIN_LLM_MODELS[self.domain])}")

        completed_ids = self.get_completed_configs()
        remaining = [c for c in self.all_configs if get_config_id(c) not in completed_ids]

        print(f"  Completed:      {len(completed_ids)}")
        print(f"  Remaining:      {len(remaining)}")
        print("=" * 70)

        if not remaining:
            print("\nâœ… All configurations completed!")
            self._save_all_results()
            return

        results = []
        try:
            for i, config in enumerate(remaining):
                config_idx = self.config_to_index.get(config, 0)
                print(f"\n[{i + 1}/{len(remaining)}] Config #{config_idx:03d}", end="")
                result = self.evaluate_config(config)
                if result:
                    results.append(result)
                self._save_all_results()

        except RateLimitExit as e:
            print(f"\n\nðŸ›‘ RATE LIMIT - EXITING")
            self._save_all_results()
            print(f"ðŸ’¾ All progress saved.")
            print(f"\nðŸ“‹ To resume, run the same command again:")
            print(f"   python run_preliminary_evaluation.py --domain {self.domain} --run")
            sys.exit(0)

        except KeyboardInterrupt:
            print(f"\n\nâš ï¸ INTERRUPTED BY USER")
            print(f"ðŸ’¾ Saving all results before exit...")
            self._save_all_results()
            print(f"\nðŸ“‹ To resume, run the same command again.")
            sys.exit(0)

        print(f"\nâœ… Evaluation complete!")
        self._save_all_results()

    def _save_all_results(self):
        results = []

        for config in self.all_configs:
            checkpoint = self._load_checkpoint(config)

            if len(checkpoint.get("completed", [])) >= NUM_EXAMPLES:
                example_results = checkpoint["results"]
                valid = [r for r in example_results if r.get("is_valid", False)]

                if valid:
                    avg_rel = float(np.mean([r["computed_relevance"] for r in valid]))
                    avg_util = float(np.mean([r["computed_utilization"] for r in valid]))
                    avg_comp = float(np.mean([r["computed_completeness"] for r in valid]))
                    avg_adh = float(np.mean([r["computed_adherence"] for r in valid]))
                    avg_trace = float(np.mean([avg_rel, avg_util, avg_comp, avg_adh]))

                    gt_avg_rel = float(np.mean([r["gt_relevance"] for r in valid]))
                    gt_avg_util = float(np.mean([r["gt_utilization"] for r in valid]))
                    gt_avg_comp = float(np.mean([r["gt_completeness"] for r in valid]))
                    gt_avg_adh = float(np.mean([r["gt_adherence"] for r in valid]))

                    rmse_rel = float(np.sqrt(np.mean([(r["computed_relevance"] - r["gt_relevance"]) ** 2 for r in valid])))
                    rmse_util = float(np.sqrt(np.mean([(r["computed_utilization"] - r["gt_utilization"]) ** 2 for r in valid])))
                    rmse_comp = float(np.sqrt(np.mean([(r["computed_completeness"] - r["gt_completeness"]) ** 2 for r in valid])))

                    binary_preds = [r["binary_pred"] for r in valid]
                    binary_gts = [r["binary_gt"] for r in valid]
                    adherence_scores = [r["computed_adherence"] for r in valid]

                    tp = sum(1 for p, g in zip(binary_preds, binary_gts) if p == 1 and g == 1)
                    fp = sum(1 for p, g in zip(binary_preds, binary_gts) if p == 1 and g == 0)
                    fn = sum(1 for p, g in zip(binary_preds, binary_gts) if p == 0 and g == 1)

                    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

                    try:
                        auroc = float(roc_auc_score(binary_gts, adherence_scores)) if len(set(binary_gts)) > 1 else 0.0
                    except:
                        auroc = 0.0

                    results.append({
                        "config_index": self.config_to_index.get(config, 0),
                        "config_type": config[0],
                        "chunking_strategy": config[1],
                        "retrieval_method": config[2],
                        "llm_model": config[3],
                        "avg_relevance": avg_rel, "avg_utilization": avg_util,
                        "avg_completeness": avg_comp, "avg_adherence": avg_adh, "avg_trace": avg_trace,
                        "gt_avg_relevance": gt_avg_rel, "gt_avg_utilization": gt_avg_util,
                        "gt_avg_completeness": gt_avg_comp, "gt_avg_adherence": gt_avg_adh,
                        "rmse_relevance": rmse_rel, "rmse_utilization": rmse_util, "rmse_completeness": rmse_comp,
                        "auroc": auroc, "f1": f1, "precision": precision, "recall": recall,
                        "num_examples": len(example_results), "num_valid": len(valid),
                        "num_failed": len(example_results) - len(valid),
                        "example_results": example_results
                    })

        results.sort(key=lambda x: x["avg_trace"], reverse=True)

        results_file = os.path.join(self.results_dir, "preliminary_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "domain": self.domain,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "num_configs": len(self.all_configs),
                "examples_per_config": NUM_EXAMPLES,
                "completed_configs": len(results),
                "results": results
            }, f, indent=2)

    def show_progress(self):
        completed = self.get_completed_configs()
        total = len(self.all_configs)

        partial = []
        total_examples_done = len(completed) * NUM_EXAMPLES

        for config in self.all_configs:
            config_id = get_config_id(config)
            if config_id in completed:
                continue
            checkpoint = self._load_checkpoint(config)
            num_done = len(checkpoint.get("completed", []))
            if num_done > 0:
                partial.append((self.config_to_index.get(config, 0), get_config_id(config), num_done))
                total_examples_done += num_done

        print(f"\nðŸ“Š Progress for {self.domain.upper()}")
        print("=" * 60)
        print(f"  Total configs:     {total}")
        print(f"  Fully completed:   {len(completed)} ({100*len(completed)/total:.1f}%)")
        print(f"  Partially done:    {len(partial)}")
        print(f"  Not started:       {total - len(completed) - len(partial)}")
        print(f"\n  Total examples:    {total * NUM_EXAMPLES}")
        print(f"  Completed evals:   {total_examples_done}")

        if partial:
            print(f"\n  Partially completed configs:")
            for config_idx, config_id, num_done in partial[:10]:
                print(f"    - [{config_idx:03d}] {config_id}: {num_done}/{NUM_EXAMPLES}")
            if len(partial) > 10:
                print(f"    ... and {len(partial) - 10} more")

        if completed:
            print(f"\n  Last 5 fully completed:")
            for cid in list(completed)[-5:]:
                print(f"    âœ” {cid}")

    def show_results(self):
        results_file = os.path.join(self.results_dir, "preliminary_results.json")
        if not os.path.exists(results_file):
            print("âŒ No results found. Run evaluation first.")
            return

        with open(results_file, 'r') as f:
            data = json.load(f)

        results = data["results"]

        print(f"\nðŸ“Š Results for {self.domain.upper()}")
        print(f"  Total: {len(results)} configurations")

        print("\n" + "=" * 115)
        print(f"{'Rank':<5} {'#':<5} {'Type':<8} {'Chunk':<10} {'Method':<18} {'LLM':<22} {'TRACe':<8} {'R':<6} {'U':<6} {'C':<6} {'A':<6}")
        print("=" * 115)

        for i, r in enumerate(results[:20], 1):
            llm_short = r["llm_model"].split("/")[-1][:20]
            method_short = r["retrieval_method"][:16]
            config_idx = r.get("config_index", 0)
            print(f"{i:<5} {config_idx:03d}   {r['config_type']:<8} {r['chunking_strategy']:<10} {method_short:<18} {llm_short:<22} "
                  f"{r['avg_trace']:.4f}  {r['avg_relevance']:.3f}  {r['avg_utilization']:.3f}  "
                  f"{r['avg_completeness']:.3f}  {r['avg_adherence']:.3f}")

        print("\n" + "=" * 105)
        print("RMSE Scores (lower is better) & Adherence Classification")
        print("=" * 105)
        print(f"{'Rank':<5} {'#':<5} {'Type':<8} {'Chunk':<10} {'Method':<18} {'LLM':<22} {'RMSE_R':<8} {'RMSE_U':<8} {'RMSE_C':<8} {'Adh_F1':<8}")
        print("-" * 105)

        for i, r in enumerate(results[:20], 1):
            llm_short = r["llm_model"].split("/")[-1][:20]
            method_short = r["retrieval_method"][:16]
            config_idx = r.get("config_index", 0)
            print(f"{i:<5} {config_idx:03d}   {r['config_type']:<8} {r['chunking_strategy']:<10} {method_short:<18} {llm_short:<22} "
                  f"{r.get('rmse_relevance', 0):.4f}  {r.get('rmse_utilization', 0):.4f}  "
                  f"{r.get('rmse_completeness', 0):.4f}  {r.get('f1', 0):.4f}")

    def show_top10(self):
        results_file = os.path.join(self.results_dir, "preliminary_results.json")
        if not os.path.exists(results_file):
            print("âŒ No results found. Run evaluation first.")
            return

        with open(results_file, 'r') as f:
            data = json.load(f)

        results = data["results"][:10]

        print(f"\n# Top 10 configurations for {self.domain}")
        print(f"# Copy this to DOMAIN_CONFIGS in run_multidomain_evaluation.py\n")
        print(f'    "{self.domain}": [')
        for r in results:
            print(f'        ("{r["config_type"]}", "{r["chunking_strategy"]}", "{r["retrieval_method"]}", "{r["llm_model"]}"),')
        print("    ],")


def list_all_domains():
    print("\nðŸ“‹ Domain Configuration Counts")
    print("=" * 70)

    total_configs = 0
    total_evals = 0

    for domain in DOMAINS:
        configs = generate_configs_for_domain(domain)
        embeddings = DOMAIN_EMBEDDINGS.get(domain, [])
        llms = DOMAIN_LLM_MODELS.get(domain, [])
        evals = len(configs) * NUM_EXAMPLES
        total_configs += len(configs)
        total_evals += evals

        print(f"\n  {domain.upper()}")
        print(f"    Configs:    {len(configs)}")
        print(f"    Embeddings: {len(embeddings)} - {', '.join(embeddings[:4])}{'...' if len(embeddings) > 4 else ''}")
        print(f"    LLMs:       {len(llms)} - {', '.join([l.split('/')[-1][:15] for l in llms[:3]])}{'...' if len(llms) > 3 else ''}")
        print(f"    Evals:      {evals}")

    print(f"\n{'='*70}")
    print(f"  TOTAL: {total_configs} configs, {total_evals} evaluations")


def main():
    parser = argparse.ArgumentParser(description="Preliminary Evaluation v10")
    parser.add_argument("--domain", type=str, choices=DOMAINS, help="Domain to evaluate")
    parser.add_argument("--list", action="store_true", help="List all domains and config counts")
    parser.add_argument("--run", action="store_true", help="Run evaluation")
    parser.add_argument("--progress", action="store_true", help="Show progress")
    parser.add_argument("--results", action="store_true", help="Show results")
    parser.add_argument("--top10", action="store_true", help="Show top 10 configs")

    args = parser.parse_args()

    if args.list:
        list_all_domains()
        return

    if not args.domain:
        parser.print_help()
        return

    evaluator = PreliminaryEvaluator(args.domain)

    if args.progress:
        evaluator.show_progress()
    elif args.results:
        evaluator.show_results()
    elif args.top10:
        evaluator.show_top10()
    elif args.run:
        evaluator.load_data()
        evaluator.initialize_judge()
        evaluator.initialize_api()
        evaluator.run_evaluation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()