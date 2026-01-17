"""
RGB (Retrieval-Augmented Generation Benchmark) Evaluator
=========================================================
Evaluates 4 fundamental LLM abilities for RAG:
1. Noise Robustness - Can LLM extract correct answers despite noisy documents?
2. Negative Rejection - Can LLM refuse to answer when documents lack the answer?
3. Information Integration - Can LLM combine info from multiple documents?
4. Counterfactual Robustness - Can LLM detect factual errors in documents?

Features:
- Checkpoint saving every 10 samples (intra-evaluation)
- Resume capability from exact sample position
- 10-second delay between LLM calls to avoid rate limits
- Progress tracking with ETA
- Graceful exit on rate limits
- Full document and response storage (no truncation)

Usage:
    python rgb_evaluator.py --all                              # Full evaluation
    python rgb_evaluator.py --all --sample_size 20             # Quick test
    python rgb_evaluator.py --all --resume                     # Resume from checkpoint
"""

import os
import json
import random
import argparse
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/rgb")
RESULTS_DIR = Path("results/rgb")
CHECKPOINT_DIR = Path("checkpoints/rgb")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = CHECKPOINT_DIR / "rgb_checkpoint.json"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Delay between LLM calls (seconds) to avoid rate limits
LLM_CALL_DELAY = 2

# Save checkpoint every N samples
CHECKPOINT_INTERVAL = 10

# System prompt from Figure 3 of the RGB paper
SYSTEM_PROMPT = """You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate 'I can not answer the question because of the insufficient information in documents.' If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer."""

# 3 LLM Models to evaluate via Groq API
LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

# Number of documents to provide per question (as per paper)
NUM_DOCS = 5

# Noise ratios to test for different abilities
NOISE_RATIOS = [0, 0.2, 0.4, 0.6, 0.8]           # For Noise Robustness
INTEGRATION_NOISE_RATIOS = [0, 0.2, 0.4]          # For Information Integration


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EvaluationResult:
    """Stores results for one evaluation run."""
    ability: str
    model: str
    metrics: Dict[str, float]
    details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "ability": self.ability,
            "model": self.model,
            "metrics": self.metrics,
            "details": self.details
        }


@dataclass
class ProgressTracker:
    """Tracks evaluation progress and estimates time remaining."""
    total_evaluations: int
    completed_evaluations: int = 0
    total_api_calls: int = 0
    completed_api_calls: int = 0
    start_time: float = field(default_factory=time.time)
    eval_times: List[float] = field(default_factory=list)

    def update(self, eval_time: float, api_calls: int):
        self.completed_evaluations += 1
        self.completed_api_calls += api_calls
        self.eval_times.append(eval_time)

    def get_eta(self) -> str:
        if not self.eval_times:
            return "Calculating..."

        avg_time = sum(self.eval_times) / len(self.eval_times)
        remaining = self.total_evaluations - self.completed_evaluations
        eta_seconds = avg_time * remaining

        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}m"
        else:
            return f"{eta_seconds/3600:.1f}h"

    def get_progress_str(self) -> str:
        pct = (self.completed_evaluations / self.total_evaluations) * 100
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        return f"[{self.completed_evaluations}/{self.total_evaluations}] {pct:.1f}% | Elapsed: {elapsed_str} | ETA: {self.get_eta()}"


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def generate_eval_key(ability: str, model: str, noise_ratio: float = None) -> str:
    """Generate unique key for each evaluation."""
    if noise_ratio is not None:
        return f"{ability}|{model}|{noise_ratio}"
    return f"{ability}|{model}"


def load_checkpoint() -> Dict:
    """Load checkpoint from file."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        "completed": [],
        "results": [],
        "in_progress": None,
        "in_progress_index": 0,
        "in_progress_details": [],
        "in_progress_counters": {},
        "timestamp": None,
        "sample_size": None
    }


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to file."""
    checkpoint["timestamp"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def clear_checkpoint():
    """Clear checkpoint file."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("   🗑️ Checkpoint cleared")


def is_completed(checkpoint: Dict, eval_key: str) -> bool:
    """Check if evaluation is already completed."""
    return eval_key in checkpoint["completed"]


def get_resume_index(checkpoint: Dict, eval_key: str) -> Tuple[int, List[Dict], Dict]:
    """Get resume index and partial data for an evaluation."""
    if checkpoint.get("in_progress") == eval_key:
        return (
            checkpoint.get("in_progress_index", 0),
            checkpoint.get("in_progress_details", []),
            checkpoint.get("in_progress_counters", {})
        )
    return 0, [], {}


def save_intra_checkpoint(checkpoint: Dict, eval_key: str, index: int, details: List[Dict], counters: Dict):
    """Save checkpoint within an evaluation (every N samples)."""
    checkpoint["in_progress"] = eval_key
    checkpoint["in_progress_index"] = index
    checkpoint["in_progress_details"] = details
    checkpoint["in_progress_counters"] = counters
    save_checkpoint(checkpoint)
    print(f"      💾 Checkpoint saved at sample {index}")


def clear_intra_progress(checkpoint: Dict):
    """Clear in-progress data after completing an evaluation."""
    checkpoint["in_progress"] = None
    checkpoint["in_progress_index"] = 0
    checkpoint["in_progress_details"] = []
    checkpoint["in_progress_counters"] = {}


# ============================================================================
# RATE LIMIT HANDLING
# ============================================================================

class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass


def handle_rate_limit(checkpoint: Dict, eval_key: str, index: int, details: List[Dict], counters: Dict):
    """Save checkpoint and exit on rate limit."""
    print("\n" + "="*70)
    print("🚨 RATE LIMIT HIT!")
    print("="*70)

    save_intra_checkpoint(checkpoint, eval_key, index, details, counters)

    print(f"   Current evaluation: {eval_key}")
    print(f"   Progress saved at sample: {index}")
    print(f"   Completed evaluations: {len(checkpoint['completed'])}")
    print("\n📋 To resume after changing API key:")
    print("   1. Update GROQ_API_KEY in .env file")
    print("   2. Run: python rgb_evaluator.py --all")
    print("="*70)
    sys.exit(1)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                record = json.loads(line)
                record['id'] = i
                data.append(record)
    return data


def load_datasets() -> Dict[str, List[Dict]]:
    """Load all three RGB datasets."""
    print("📂 Loading datasets...")
    datasets = {
        "refine": load_jsonl(DATA_DIR / "en_refine.json"),
        "integration": load_jsonl(DATA_DIR / "en_int.json"),
        "fact": load_jsonl(DATA_DIR / "en_fact.json"),
    }
    print(f"   ✓ refine: {len(datasets['refine'])} records")
    print(f"   ✓ integration: {len(datasets['integration'])} records")
    print(f"   ✓ fact: {len(datasets['fact'])} records")
    return datasets


# ============================================================================
# DOCUMENT SELECTION STRATEGIES
# ============================================================================

def select_documents_noise_robustness(
    record: Dict,
    noise_ratio: float,
    num_docs: int = NUM_DOCS
) -> Tuple[List[str], List[str], List[str]]:
    """
    Select documents for noise robustness evaluation.

    Returns:
        Tuple of (combined_docs, positive_docs_used, negative_docs_used)
    """
    positive_docs = record.get("positive", [])
    negative_docs = record.get("negative", [])

    num_noise = int(num_docs * noise_ratio)
    num_positive = num_docs - num_noise

    selected_positive = random.sample(positive_docs, min(num_positive, len(positive_docs)))
    selected_negative = random.sample(negative_docs, min(num_noise, len(negative_docs)))

    docs = selected_positive + selected_negative
    random.shuffle(docs)

    return docs, selected_positive, selected_negative


def select_documents_negative_rejection(
    record: Dict,
    num_docs: int = NUM_DOCS
) -> Tuple[List[str], List[str]]:
    """
    Select ONLY negative documents for rejection testing.

    Returns:
        Tuple of (selected_negative_docs, all for reference)
    """
    negative_docs = record.get("negative", [])
    selected = random.sample(negative_docs, min(num_docs, len(negative_docs)))
    return selected, selected


def select_documents_information_integration(
    record: Dict,
    noise_ratio: float,
    num_docs: int = NUM_DOCS
) -> Tuple[List[str], List[str], List[str]]:
    """
    Select documents for information integration.

    Returns:
        Tuple of (combined_docs, positive_docs_used, negative_docs_used)
    """
    positive_docs = []
    pos_data = record.get("positive", [])
    if isinstance(pos_data, list):
        for item in pos_data:
            if isinstance(item, list):
                positive_docs.extend(item)
            else:
                positive_docs.append(item)

    negative_docs = record.get("negative", [])

    num_noise = int(num_docs * noise_ratio)
    num_positive = num_docs - num_noise

    selected_positive = random.sample(positive_docs, min(num_positive, len(positive_docs)))
    selected_negative = random.sample(negative_docs, min(num_noise, len(negative_docs))) if negative_docs else []

    docs = selected_positive + selected_negative
    random.shuffle(docs)

    return docs, selected_positive, selected_negative


def select_documents_counterfactual(
    record: Dict,
    use_wrong: bool = True
) -> Tuple[List[str], str]:
    """
    Select documents for counterfactual robustness.

    Returns:
        Tuple of (docs, doc_type) where doc_type is 'positive_wrong' or 'positive'
    """
    doc_type = "positive_wrong" if use_wrong else "positive"
    docs = record.get(doc_type, [])

    if isinstance(docs, list) and len(docs) > 0:
        return docs[:NUM_DOCS], doc_type
    return [], doc_type


# ============================================================================
# LLM INTERACTION
# ============================================================================

def call_llm(client: Groq, model: str, query: str, documents: List[str],
             checkpoint: Dict, eval_key: str, index: int, details: List[Dict], counters: Dict) -> str:
    """Call LLM with query and documents. Handles rate limits with checkpoint save."""
    docs_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])

    user_prompt = f"""External Documents:
{docs_text}

Question: {query}

Please answer the question based on the provided documents."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=512
        )
        time.sleep(LLM_CALL_DELAY)
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e).lower()
        print(f"      ⚠️ API Error: {e}")
        if "rate" in error_str or "limit" in error_str or "429" in error_str or "quota" in error_str:
            handle_rate_limit(checkpoint, eval_key, index, details, counters)
        raise


def call_llm_no_docs(client: Groq, model: str, query: str,
                     checkpoint: Dict, eval_key: str, index: int, details: List[Dict], counters: Dict) -> str:
    """Call LLM without any documents (for counterfactual baseline)."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Answer this question: {query}"}
            ],
            temperature=0,
            max_tokens=512
        )
        time.sleep(LLM_CALL_DELAY)
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e).lower()
        if "rate" in error_str or "limit" in error_str or "429" in error_str or "quota" in error_str:
            handle_rate_limit(checkpoint, eval_key, index, details, counters)
        print(f"      ⚠️ API Error: {e}")
        raise


# ============================================================================
# ANSWER CHECKING
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison - handles Unicode and formatting issues."""
    import re
    import unicodedata

    text = text.lower()
    text = unicodedata.normalize('NFKC', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Replace various dash types with regular dash
    text = re.sub(r'[–—−]', '-', text)

    # Replace various quote types with regular quotes
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"[''`]", "'", text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def check_accuracy(response: str, answer: Any) -> bool:
    """Check if response contains the correct answer."""
    response_normalized = normalize_text(response)

    def flatten_and_check(ans) -> bool:
        if isinstance(ans, list):
            return any(flatten_and_check(item) for item in ans)
        ans_normalized = normalize_text(str(ans))
        return ans_normalized in response_normalized

    return flatten_and_check(answer)


def check_rejection(response: str) -> bool:
    """Check if LLM properly rejected answering."""
    rejection_phrases = [
        "i can not answer",
        "i cannot answer",
        "cannot answer",
        "can't answer",
        "insufficient information",
        "not enough information",
        "no relevant information",
        "documents do not contain",
        "unable to answer",
        "don't have enough",
        "do not have enough"
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in rejection_phrases)


def check_error_detection(response: str) -> bool:
    """Check if LLM detected factual errors."""
    error_phrases = [
        "factual error",
        "factual errors",
        "incorrect information",
        "wrong information",
        "inaccurate",
        "not accurate",
        "error in the document",
        "inconsistencies",
        "contradicts"
    ]
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in error_phrases)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_noise_robustness(
    client: Groq,
    model: str,
    data: List[Dict],
    noise_ratio: float,
    sample_size: int,
    checkpoint: Dict
) -> Tuple[EvaluationResult, int]:
    """Evaluate noise robustness with full document storage."""
    eval_key = generate_eval_key("noise_robustness", model, noise_ratio)
    print(f"\n  📊 Noise Ratio: {noise_ratio}")

    if sample_size:
        data = data[:sample_size]

    total = len(data)
    start_index, details, counters = get_resume_index(checkpoint, eval_key)
    correct = counters.get("correct", 0)

    if start_index > 0:
        print(f"      ⏩ Resuming from sample {start_index}/{total}")

    api_calls = 0

    for i in range(start_index, total):
        record = data[i]

        # Get documents with tracking of positive/negative
        docs, positive_used, negative_used = select_documents_noise_robustness(record, noise_ratio)

        response = call_llm(client, model, record["query"], docs,
                           checkpoint, eval_key, i, details, {"correct": correct})
        api_calls += 1

        is_correct = check_accuracy(response, record["answer"])
        if is_correct:
            correct += 1

        # Store full details without truncation
        details.append({
            "id": record["id"],
            "query": record["query"],
            "expected_answer": record["answer"],
            "correct": is_correct,
            "response": response,  # Full response, no truncation
            "documents_provided": docs,  # All documents sent to LLM
            "positive_documents_used": positive_used,  # Which were positive
            "negative_documents_used": negative_used,  # Which were negative (noise)
            "noise_ratio": noise_ratio,
            "num_positive": len(positive_used),
            "num_negative": len(negative_used)
        })

        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{total} | Accuracy: {correct/(i+1)*100:.1f}%")

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_intra_checkpoint(checkpoint, eval_key, i + 1, details, {"correct": correct})

    accuracy = (correct / total) * 100
    print(f"      ✅ Final Accuracy: {accuracy:.2f}%")

    clear_intra_progress(checkpoint)

    return EvaluationResult(
        ability="noise_robustness",
        model=model,
        metrics={"noise_ratio": noise_ratio, "accuracy": accuracy},
        details=details
    ), api_calls


def evaluate_negative_rejection(
    client: Groq,
    model: str,
    data: List[Dict],
    sample_size: int,
    checkpoint: Dict
) -> Tuple[EvaluationResult, int]:
    """Evaluate negative rejection with full document storage."""
    eval_key = generate_eval_key("negative_rejection", model)
    print(f"\n  📊 Evaluating Negative Rejection")

    if sample_size:
        data = data[:sample_size]

    total = len(data)
    start_index, details, counters = get_resume_index(checkpoint, eval_key)
    rejections = counters.get("rejections", 0)

    if start_index > 0:
        print(f"      ⏩ Resuming from sample {start_index}/{total}")

    api_calls = 0

    for i in range(start_index, total):
        record = data[i]

        # Get only negative documents
        docs, negative_used = select_documents_negative_rejection(record)

        response = call_llm(client, model, record["query"], docs,
                           checkpoint, eval_key, i, details, {"rejections": rejections})
        api_calls += 1

        is_rejected = check_rejection(response)
        if is_rejected:
            rejections += 1

        # Store full details without truncation
        details.append({
            "id": record["id"],
            "query": record["query"],
            "expected_answer": record["answer"],  # The answer that's NOT in the docs
            "rejected": is_rejected,
            "response": response,  # Full response, no truncation
            "documents_provided": docs,  # All negative documents sent to LLM
            "negative_documents_used": negative_used,
            "num_documents": len(docs),
            "all_positive_in_dataset": record.get("positive", []),  # For reference
            "all_negative_in_dataset": record.get("negative", [])   # For reference
        })

        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{total} | Rejection Rate: {rejections/(i+1)*100:.1f}%")

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_intra_checkpoint(checkpoint, eval_key, i + 1, details, {"rejections": rejections})

    rejection_rate = (rejections / total) * 100
    print(f"      ✅ Final Rejection Rate: {rejection_rate:.2f}%")

    clear_intra_progress(checkpoint)

    return EvaluationResult(
        ability="negative_rejection",
        model=model,
        metrics={"rejection_rate": rejection_rate},
        details=details
    ), api_calls


def evaluate_information_integration(
    client: Groq,
    model: str,
    data: List[Dict],
    noise_ratio: float,
    sample_size: int,
    checkpoint: Dict
) -> Tuple[EvaluationResult, int]:
    """Evaluate information integration with full document storage."""
    eval_key = generate_eval_key("information_integration", model, noise_ratio)
    print(f"\n  📊 Info Integration (noise_ratio={noise_ratio})")

    if sample_size:
        data = data[:sample_size]

    total = len(data)
    start_index, details, counters = get_resume_index(checkpoint, eval_key)
    correct = counters.get("correct", 0)

    if start_index > 0:
        print(f"      ⏩ Resuming from sample {start_index}/{total}")

    api_calls = 0

    for i in range(start_index, total):
        record = data[i]

        # Get documents with tracking
        docs, positive_used, negative_used = select_documents_information_integration(record, noise_ratio)

        response = call_llm(client, model, record["query"], docs,
                           checkpoint, eval_key, i, details, {"correct": correct})
        api_calls += 1

        is_correct = check_accuracy(response, record["answer"])
        if is_correct:
            correct += 1

        # Store full details without truncation
        details.append({
            "id": record["id"],
            "query": record["query"],
            "expected_answer": record["answer"],
            "correct": is_correct,
            "response": response,  # Full response, no truncation
            "documents_provided": docs,  # All documents sent to LLM
            "positive_documents_used": positive_used,
            "negative_documents_used": negative_used,
            "noise_ratio": noise_ratio,
            "num_positive": len(positive_used),
            "num_negative": len(negative_used),
            "all_positive_in_dataset": record.get("positive", []),  # Original structure
            "all_negative_in_dataset": record.get("negative", [])
        })

        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{total} | Accuracy: {correct/(i+1)*100:.1f}%")

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_intra_checkpoint(checkpoint, eval_key, i + 1, details, {"correct": correct})

    accuracy = (correct / total) * 100
    print(f"      ✅ Final Accuracy: {accuracy:.2f}%")

    clear_intra_progress(checkpoint)

    return EvaluationResult(
        ability="information_integration",
        model=model,
        metrics={"noise_ratio": noise_ratio, "accuracy": accuracy},
        details=details
    ), api_calls


def evaluate_counterfactual_robustness(
    client: Groq,
    model: str,
    data: List[Dict],
    sample_size: int,
    checkpoint: Dict
) -> Tuple[EvaluationResult, int]:
    """Evaluate counterfactual robustness with full document storage."""
    eval_key = generate_eval_key("counterfactual_robustness", model)
    print(f"\n  📊 Evaluating Counterfactual Robustness")

    if sample_size:
        data = data[:sample_size]

    total = len(data)
    start_index, details, counters = get_resume_index(checkpoint, eval_key)
    acc_no_docs = counters.get("acc_no_docs", 0)
    acc_with_docs = counters.get("acc_with_docs", 0)
    error_detected = counters.get("error_detected", 0)
    error_corrected = counters.get("error_corrected", 0)

    if start_index > 0:
        print(f"      ⏩ Resuming from sample {start_index}/{total}")

    api_calls = 0

    for i in range(start_index, total):
        record = data[i]
        query = record["query"]
        correct_answer = record["answer"]
        fake_answer = record.get("fakeanswer", None)

        current_counters = {
            "acc_no_docs": acc_no_docs,
            "acc_with_docs": acc_with_docs,
            "error_detected": error_detected,
            "error_corrected": error_corrected
        }

        # 1. Test without documents (baseline)
        response_no_docs = call_llm_no_docs(client, model, query,
                                            checkpoint, eval_key, i, details, current_counters)
        api_calls += 1
        has_correct_no_docs = check_accuracy(response_no_docs, correct_answer)
        if has_correct_no_docs:
            acc_no_docs += 1

        current_counters["acc_no_docs"] = acc_no_docs

        # 2. Test with counterfactual (wrong) documents
        docs_wrong, doc_type_wrong = select_documents_counterfactual(record, use_wrong=True)
        response_with_docs = call_llm(client, model, query, docs_wrong,
                                      checkpoint, eval_key, i, details, current_counters)
        api_calls += 1

        has_correct_with_docs = check_accuracy(response_with_docs, correct_answer)
        if has_correct_with_docs:
            acc_with_docs += 1

        detected_error = check_error_detection(response_with_docs)
        if detected_error:
            error_detected += 1

        if detected_error and has_correct_with_docs:
            error_corrected += 1

        # Also get correct documents for reference
        docs_correct, doc_type_correct = select_documents_counterfactual(record, use_wrong=False)

        # Store full details without truncation
        details.append({
            "id": record["id"],
            "query": query,
            "correct_answer": str(correct_answer),
            "fake_answer": str(fake_answer) if fake_answer else None,
            "acc_no_docs": has_correct_no_docs,
            "acc_with_docs": has_correct_with_docs,
            "error_detected": detected_error,
            "error_corrected": detected_error and has_correct_with_docs,
            "response_no_docs": response_no_docs,  # Full response
            "response_with_wrong_docs": response_with_docs,  # Full response
            "wrong_documents_provided": docs_wrong,  # Counterfactual docs sent to LLM
            "correct_documents_reference": docs_correct,  # Correct docs for comparison
            "all_positive_wrong_in_dataset": record.get("positive_wrong", []),
            "all_positive_in_dataset": record.get("positive", [])
        })

        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{total}")

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_intra_checkpoint(checkpoint, eval_key, i + 1, details, {
                "acc_no_docs": acc_no_docs,
                "acc_with_docs": acc_with_docs,
                "error_detected": error_detected,
                "error_corrected": error_corrected
            })

    acc_pct = (acc_no_docs / total) * 100
    acc_doc_pct = (acc_with_docs / total) * 100
    ed_pct = (error_detected / total) * 100
    cr_pct = (error_corrected / error_detected * 100) if error_detected > 0 else 0

    print(f"      ✅ Acc (no docs): {acc_pct:.2f}%")
    print(f"      ✅ Acc (with counterfactual): {acc_doc_pct:.2f}%")
    print(f"      ✅ Error Detection: {ed_pct:.2f}%")
    print(f"      ✅ Error Correction: {cr_pct:.2f}%")

    clear_intra_progress(checkpoint)

    return EvaluationResult(
        ability="counterfactual_robustness",
        model=model,
        metrics={
            "acc": acc_pct,
            "acc_doc": acc_doc_pct,
            "error_detection": ed_pct,
            "error_correction": cr_pct
        },
        details=details
    ), api_calls


# ============================================================================
# RESULTS FORMATTING
# ============================================================================

def print_summary_tables(results: List[Dict]):
    """Print results in paper-style tables."""

    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY (Paper Format)")
    print("="*80)

    noise_results = [r for r in results if r["ability"] == "noise_robustness"]
    rejection_results = [r for r in results if r["ability"] == "negative_rejection"]
    integration_results = [r for r in results if r["ability"] == "information_integration"]
    counterfactual_results = [r for r in results if r["ability"] == "counterfactual_robustness"]

    # Table 1: Noise Robustness
    print("\n📈 TABLE 1: Noise Robustness (Accuracy %)")
    print("-" * 75)
    header = f"{'Model':<40} | " + " | ".join([f"{r:>5}" for r in NOISE_RATIOS])
    print(header)
    print("-" * 75)

    for model in LLM_MODELS:
        model_results = [r for r in noise_results if r["model"] == model]
        if model_results:
            row = f"{model:<40} | "
            for nr in NOISE_RATIOS:
                match = next((r for r in model_results if r["metrics"].get("noise_ratio") == nr), None)
                if match:
                    row += f"{match['metrics']['accuracy']:>5.1f} | "
                else:
                    row += "  N/A | "
            print(row)

    # Table 2: Negative Rejection
    print("\n📈 TABLE 2: Negative Rejection (Rejection Rate %)")
    print("-" * 55)
    print(f"{'Model':<40} | {'Rate':>10}")
    print("-" * 55)
    for r in rejection_results:
        print(f"{r['model']:<40} | {r['metrics']['rejection_rate']:>10.1f}%")

    # Table 3: Information Integration
    print("\n📈 TABLE 3: Information Integration (Accuracy %)")
    print("-" * 65)
    header = f"{'Model':<40} | " + " | ".join([f"{r:>5}" for r in INTEGRATION_NOISE_RATIOS])
    print(header)
    print("-" * 65)

    for model in LLM_MODELS:
        model_results = [r for r in integration_results if r["model"] == model]
        if model_results:
            row = f"{model:<40} | "
            for nr in INTEGRATION_NOISE_RATIOS:
                match = next((r for r in model_results if r["metrics"].get("noise_ratio") == nr), None)
                if match:
                    row += f"{match['metrics']['accuracy']:>5.1f} | "
                else:
                    row += "  N/A | "
            print(row)

    # Table 4: Counterfactual Robustness
    print("\n📈 TABLE 4: Counterfactual Robustness (%)")
    print("-" * 80)
    print(f"{'Model':<40} | {'Acc':>6} | {'Acc_doc':>7} | {'ED':>6} | {'CR':>6}")
    print("-" * 80)
    for r in counterfactual_results:
        m = r["metrics"]
        print(f"{r['model']:<40} | {m['acc']:>6.1f} | {m['acc_doc']:>7.1f} | {m['error_detection']:>6.1f} | {m['error_correction']:>6.1f}")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def calculate_total_evaluations() -> int:
    """Calculate total number of evaluation runs."""
    return len(LLM_MODELS) * 10


def run_full_evaluation(sample_size: int = None):
    """Run complete evaluation across all abilities and models."""

    checkpoint = load_checkpoint()

    if checkpoint["completed"] or checkpoint.get("in_progress"):
        completed = len(checkpoint['completed'])
        in_prog = checkpoint.get('in_progress', 'None')
        in_prog_idx = checkpoint.get('in_progress_index', 0)
        print(f"📂 Resuming from checkpoint:")
        print(f"   Completed evaluations: {completed}")
        if in_prog:
            print(f"   In progress: {in_prog} (sample {in_prog_idx})")
    else:
        print("📂 No checkpoint found, starting fresh")
        checkpoint["sample_size"] = sample_size

    total_evals = calculate_total_evaluations()
    already_done = len(checkpoint["completed"])
    tracker = ProgressTracker(total_evaluations=total_evals, completed_evaluations=already_done)

    print("\n" + "="*70)
    print("🚀 RGB Benchmark Evaluation")
    print("="*70)
    print(f"   Models: {len(LLM_MODELS)}")
    print(f"   Sample size: {sample_size or 'Full dataset'}")
    print(f"   Total evaluations: {total_evals}")
    print(f"   Already completed: {already_done}")
    print(f"   Remaining: {total_evals - already_done}")
    print(f"   Delay between calls: {LLM_CALL_DELAY}s")
    print(f"   Checkpoint interval: Every {CHECKPOINT_INTERVAL} samples")
    print(f"   ⚠️  Full document & response storage enabled (no truncation)")
    print("="*70)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    datasets = load_datasets()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in LLM_MODELS:
        print(f"\n{'='*70}")
        print(f"🤖 Model: {model}")
        print("="*70)

        # 1. Noise Robustness
        print("\n📈 NOISE ROBUSTNESS")
        for noise_ratio in NOISE_RATIOS:
            eval_key = generate_eval_key("noise_robustness", model, noise_ratio)

            if is_completed(checkpoint, eval_key):
                print(f"  ⏭️ Skipping noise_ratio={noise_ratio} (already completed)")
                continue

            eval_start = time.time()
            result, api_calls = evaluate_noise_robustness(
                client, model, datasets["refine"],
                noise_ratio, sample_size, checkpoint
            )
            eval_time = time.time() - eval_start

            checkpoint["completed"].append(eval_key)
            checkpoint["results"].append(result.to_dict())
            save_checkpoint(checkpoint)

            tracker.update(eval_time, api_calls)
            print(f"  📊 {tracker.get_progress_str()}")

        # 2. Negative Rejection
        print("\n📈 NEGATIVE REJECTION")
        eval_key = generate_eval_key("negative_rejection", model)

        if is_completed(checkpoint, eval_key):
            print(f"  ⏭️ Skipping (already completed)")
        else:
            eval_start = time.time()
            result, api_calls = evaluate_negative_rejection(
                client, model, datasets["refine"], sample_size, checkpoint
            )
            eval_time = time.time() - eval_start

            checkpoint["completed"].append(eval_key)
            checkpoint["results"].append(result.to_dict())
            save_checkpoint(checkpoint)

            tracker.update(eval_time, api_calls)
            print(f"  📊 {tracker.get_progress_str()}")

        # 3. Information Integration
        print("\n📈 INFORMATION INTEGRATION")
        for noise_ratio in INTEGRATION_NOISE_RATIOS:
            eval_key = generate_eval_key("information_integration", model, noise_ratio)

            if is_completed(checkpoint, eval_key):
                print(f"  ⏭️ Skipping noise_ratio={noise_ratio} (already completed)")
                continue

            eval_start = time.time()
            result, api_calls = evaluate_information_integration(
                client, model, datasets["integration"],
                noise_ratio, sample_size, checkpoint
            )
            eval_time = time.time() - eval_start

            checkpoint["completed"].append(eval_key)
            checkpoint["results"].append(result.to_dict())
            save_checkpoint(checkpoint)

            tracker.update(eval_time, api_calls)
            print(f"  📊 {tracker.get_progress_str()}")

        # 4. Counterfactual Robustness
        print("\n📈 COUNTERFACTUAL ROBUSTNESS")
        eval_key = generate_eval_key("counterfactual_robustness", model)

        if is_completed(checkpoint, eval_key):
            print(f"  ⏭️ Skipping (already completed)")
        else:
            eval_start = time.time()
            result, api_calls = evaluate_counterfactual_robustness(
                client, model, datasets["fact"], sample_size, checkpoint
            )
            eval_time = time.time() - eval_start

            checkpoint["completed"].append(eval_key)
            checkpoint["results"].append(result.to_dict())
            save_checkpoint(checkpoint)

            tracker.update(eval_time, api_calls)
            print(f"  📊 {tracker.get_progress_str()}")

    # Save final results
    results_file = RESULTS_DIR / f"rgb_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(checkpoint["results"], f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    print_summary_tables(checkpoint["results"])

    print(f"\n💾 Checkpoint retained at: {CHECKPOINT_FILE}")
    print("\n🎉 Evaluation complete!")

    return checkpoint["results"]


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="RGB Benchmark Evaluator")
    parser.add_argument("--ability", type=str,
                        choices=["noise", "rejection", "integration", "counterfactual", "all"],
                        help="Which ability to evaluate")
    parser.add_argument("--noise_ratio", type=float, default=0.4,
                        help="Noise ratio for noise/integration tests (default: 0.4)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to test (default: first in list)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Limit samples per evaluation (for quick testing)")
    parser.add_argument("--all", action="store_true",
                        help="Run full evaluation across all models and abilities")
    parser.add_argument("--clear_checkpoint", action="store_true",
                        help="Clear existing checkpoint and start fresh")

    args = parser.parse_args()

    if args.clear_checkpoint:
        clear_checkpoint()
        print("Checkpoint cleared. Run again without --clear_checkpoint to start evaluation.")
        return

    if args.all or args.ability == "all":
        run_full_evaluation(sample_size=args.sample_size)
    elif args.ability:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        datasets = load_datasets()
        model = args.model or LLM_MODELS[0]
        checkpoint = {"completed": [], "results": [], "in_progress": None,
                     "in_progress_index": 0, "in_progress_details": [],
                     "in_progress_counters": {}}

        print(f"🤖 Testing: {model}")

        if args.ability == "noise":
            evaluate_noise_robustness(client, model, datasets["refine"],
                                      args.noise_ratio, args.sample_size, checkpoint)
        elif args.ability == "rejection":
            evaluate_negative_rejection(client, model, datasets["refine"],
                                        args.sample_size, checkpoint)
        elif args.ability == "integration":
            evaluate_information_integration(client, model, datasets["integration"],
                                             args.noise_ratio, args.sample_size, checkpoint)
        elif args.ability == "counterfactual":
            evaluate_counterfactual_robustness(client, model, datasets["fact"],
                                               args.sample_size, checkpoint)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()