"""
RGB Validator - Debug script to inspect actual LLM responses
Helps validate why certain evaluations return 0% accuracy
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

# Configuration
DATA_DIR = Path("data/rgb")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SYSTEM_PROMPT = """You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate 'I can not answer the question because of the insufficient information in documents.' If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer."""

LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


def load_jsonl(filepath: Path):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                record = json.loads(line)
                record['id'] = i
                data.append(record)
    return data


def normalize_text(text: str) -> str:
    """Normalize text for comparison - handles Unicode and formatting issues."""
    import re
    import unicodedata

    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'[–—−]', '-', text)
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def validate_noise_robustness(client, model, record, noise_ratio=0.0):
    """Test noise robustness and show detailed output."""
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATING: {model}")
    print(f"   Ability: Noise Robustness | Noise Ratio: {noise_ratio}")
    print(f"{'='*70}")

    # Select documents
    positive_docs = record.get("positive", [])
    negative_docs = record.get("negative", [])

    num_noise = int(5 * noise_ratio)
    num_positive = 5 - num_noise

    selected_positive = positive_docs[:num_positive]
    selected_negative = negative_docs[:num_noise]
    docs = selected_positive + selected_negative

    print(f"\n📄 QUESTION:")
    print(f"   {record['query']}")

    print(f"\n📚 EXPECTED ANSWER:")
    print(f"   {record['answer']}")

    print(f"\n📑 DOCUMENTS PROVIDED ({len(docs)} total: {num_positive} positive, {num_noise} negative):")
    for i, doc in enumerate(docs[:2]):  # Show first 2 docs
        print(f"   Doc {i+1}: {doc[:150]}...")
    if len(docs) > 2:
        print(f"   ... and {len(docs)-2} more documents")

    # Call LLM
    docs_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    user_prompt = f"""External Documents:
{docs_text}

Question: {record['query']}

Please answer the question based on the provided documents."""

    print(f"\n🤖 CALLING LLM...")
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
        llm_response = response.choices[0].message.content.strip()
    except Exception as e:
        llm_response = f"ERROR: {e}"

    print(f"\n💬 LLM RESPONSE:")
    print(f"   {llm_response[:500]}")
    if len(llm_response) > 500:
        print(f"   ... (truncated, {len(llm_response)} chars total)")

    # Check accuracy
    answer = record['answer']
    response_normalized = normalize_text(llm_response)

    def flatten_and_check(ans) -> bool:
        """Recursively flatten nested lists and check each answer."""
        if isinstance(ans, list):
            return any(flatten_and_check(item) for item in ans)
        ans_normalized = normalize_text(str(ans))
        return ans_normalized in response_normalized

    is_correct = flatten_and_check(answer)

    print(f"\n✅ ACCURACY CHECK:")
    print(f"   Expected answer(s): {answer}")
    print(f"   Found in response: {'YES ✓' if is_correct else 'NO ✗'}")

    print(f"   Found in response: {'YES ✓' if is_correct else 'NO ✗'}")

    return is_correct


def validate_negative_rejection(client, model, record):
    """Test negative rejection and show detailed output."""
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATING: {model}")
    print(f"   Ability: Negative Rejection")
    print(f"{'='*70}")

    # Use only negative documents
    negative_docs = record.get("negative", [])[:5]

    print(f"\n📄 QUESTION:")
    print(f"   {record['query']}")

    print(f"\n📚 EXPECTED ANSWER (not in documents):")
    print(f"   {record['answer']}")

    print(f"\n📑 DOCUMENTS PROVIDED ({len(negative_docs)} negative/noisy docs):")
    for i, doc in enumerate(negative_docs[:2]):
        print(f"   Doc {i+1}: {doc[:150]}...")

    # Call LLM
    docs_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(negative_docs)])
    user_prompt = f"""External Documents:
{docs_text}

Question: {record['query']}

Please answer the question based on the provided documents."""

    print(f"\n🤖 CALLING LLM...")
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
        llm_response = response.choices[0].message.content.strip()
    except Exception as e:
        llm_response = f"ERROR: {e}"

    print(f"\n💬 LLM RESPONSE:")
    print(f"   {llm_response[:500]}")

    # Check rejection
    rejection_phrases = [
        "i can not answer", "i cannot answer", "cannot answer",
        "can't answer", "insufficient information", "not enough information",
        "no relevant information", "documents do not contain", "unable to answer"
    ]
    response_lower = llm_response.lower()
    is_rejected = any(phrase in response_lower for phrase in rejection_phrases)

    print(f"\n✅ REJECTION CHECK:")
    print(f"   Looking for rejection phrases...")
    print(f"   Rejected properly: {'YES ✓' if is_rejected else 'NO ✗'}")

    return is_rejected


def validate_counterfactual(client, model, record):
    """Test counterfactual robustness and show detailed output."""
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATING: {model}")
    print(f"   Ability: Counterfactual Robustness")
    print(f"{'='*70}")

    query = record['query']
    correct_answer = record['answer']
    fake_answer = record.get('fakeanswer', 'N/A')
    wrong_docs = record.get('positive_wrong', [])[:5]

    print(f"\n📄 QUESTION:")
    print(f"   {query}")

    print(f"\n📚 CORRECT ANSWER:")
    print(f"   {correct_answer}")

    print(f"\n🚫 FAKE ANSWER (in wrong docs):")
    print(f"   {fake_answer}")

    print(f"\n📑 COUNTERFACTUAL DOCUMENTS ({len(wrong_docs)} docs with wrong info):")
    for i, doc in enumerate(wrong_docs[:2]):
        print(f"   Doc {i+1}: {doc[:150]}...")

    # Call LLM with wrong docs
    docs_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(wrong_docs)])
    user_prompt = f"""External Documents:
{docs_text}

Question: {query}

Please answer the question based on the provided documents."""

    print(f"\n🤖 CALLING LLM (with counterfactual docs)...")
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
        llm_response = response.choices[0].message.content.strip()
    except Exception as e:
        llm_response = f"ERROR: {e}"

    print(f"\n💬 LLM RESPONSE:")
    print(f"   {llm_response[:500]}")

    # Check metrics
    response_lower = llm_response.lower()

    def flatten_and_check(ans) -> bool:
        """Recursively flatten nested lists and check each answer."""
        if isinstance(ans, list):
            return any(flatten_and_check(item) for item in ans)
        return str(ans).lower() in response_lower

    has_correct = flatten_and_check(correct_answer)

    error_phrases = ["factual error", "factual errors", "incorrect information",
                     "wrong information", "inaccurate", "error in the document"]
    detected_error = any(phrase in response_lower for phrase in error_phrases)

    print(f"\n✅ COUNTERFACTUAL CHECK:")
    print(f"   Contains correct answer: {'YES ✓' if has_correct else 'NO ✗'}")
    print(f"   Detected factual error: {'YES ✓' if detected_error else 'NO ✗'}")

    return has_correct, detected_error


def main():
    print("="*70)
    print("🔬 RGB VALIDATION TOOL")
    print("="*70)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load datasets
    refine_data = load_jsonl(DATA_DIR / "en_refine.json")
    fact_data = load_jsonl(DATA_DIR / "en_fact.json")

    # Use first record for validation
    test_record_refine = refine_data[0]
    test_record_fact = fact_data[0]

    print(f"\n📂 Loaded {len(refine_data)} refine records, {len(fact_data)} fact records")
    print(f"   Using record ID 0 for validation")

    # Test each model
    for model in LLM_MODELS:
        # Test 1: Noise Robustness (0% noise)
        validate_noise_robustness(client, model, test_record_refine, noise_ratio=0.0)

        # Test 2: Negative Rejection
        validate_negative_rejection(client, model, test_record_refine)

        # Test 3: Counterfactual
        validate_counterfactual(client, model, test_record_fact)

        print("\n" + "="*70)
        input("Press Enter to continue to next model...")


if __name__ == "__main__":
    main()