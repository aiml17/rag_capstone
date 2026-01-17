"""
GPT-OSS-120B Diagnostic Script
Investigates why this model shows ~35% accuracy across all tasks
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

DATA_DIR = Path("data/rgb")

SYSTEM_PROMPT = """You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate 'I can not answer the question because of the insufficient information in documents.' If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer."""


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

    # Convert to lowercase
    text = text.lower()

    # Normalize Unicode characters (NFKC normalizes special chars to ASCII equivalents)
    text = unicodedata.normalize('NFKC', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_

    # Replace various dash types with regular dash
    text = re.sub(r'[–—−]', '-', text)  # en-dash, em-dash, minus

    # Replace various quote types with regular quotes
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"[''`]", "'", text)

    # Normalize whitespace (non-breaking spaces, etc.)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def check_accuracy(response: str, answer) -> bool:
    """Check accuracy with nested list support and text normalization."""
    response_normalized = normalize_text(response)

    def flatten_and_check(ans) -> bool:
        if isinstance(ans, list):
            return any(flatten_and_check(item) for item in ans)
        ans_normalized = normalize_text(str(ans))
        return ans_normalized in response_normalized

    return flatten_and_check(answer)


def diagnose_model(client, model, data, num_samples=5):
    """Run diagnostic on a model with detailed output."""

    print(f"\n{'='*80}")
    print(f"🔬 DIAGNOSING: {model}")
    print(f"{'='*80}")

    correct_count = 0

    for i, record in enumerate(data[:num_samples]):
        print(f"\n{'─'*80}")
        print(f"📋 SAMPLE {i+1}/{num_samples}")
        print(f"{'─'*80}")

        # Get positive documents
        docs = record.get("positive", [])[:5]
        docs_text = "\n".join([f"Document {j+1}: {doc}" for j, doc in enumerate(docs)])

        user_prompt = f"""External Documents:
{docs_text}

Question: {record['query']}

Please answer the question based on the provided documents."""

        print(f"\n❓ QUESTION:")
        print(f"   {record['query']}")

        print(f"\n📚 EXPECTED ANSWER:")
        answer = record['answer']
        if isinstance(answer, list):
            # Flatten and show first few variants
            flat_answers = []
            def flatten(a):
                if isinstance(a, list):
                    for item in a:
                        flatten(item)
                else:
                    flat_answers.append(str(a))
            flatten(answer)
            print(f"   Variants: {flat_answers[:5]}...")
        else:
            print(f"   {answer}")

        # Call the model
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

        print(f"\n🤖 MODEL RESPONSE:")
        print(f"   {'-'*70}")
        # Print full response with line breaks for readability
        for line in llm_response.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*70}")
        print(f"   Response length: {len(llm_response)} chars")

        # Check accuracy
        is_correct = check_accuracy(llm_response, answer)
        correct_count += 1 if is_correct else 0

        print(f"\n✅ ACCURACY CHECK:")
        print(f"   Result: {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")

        if not is_correct:
            # Show why it failed
            print(f"\n🔍 FAILURE ANALYSIS:")
            response_lower = llm_response.lower()

            # Check if answer appears anywhere
            if isinstance(answer, list):
                for ans in flat_answers[:5]:
                    ans_lower = str(ans).lower()
                    if ans_lower in response_lower:
                        print(f"   ⚠️ Answer '{ans}' IS in response but check failed!")
                        # Find position
                        pos = response_lower.find(ans_lower)
                        print(f"   📍 Found at position {pos}: '...{llm_response[max(0,pos-20):pos+len(ans)+20]}...'")
                    else:
                        print(f"   ❌ '{ans}' not found in response")
            else:
                ans_lower = str(answer).lower()
                if ans_lower in response_lower:
                    print(f"   ⚠️ Answer IS in response but check failed!")
                else:
                    print(f"   ❌ '{answer}' not found in response")

    print(f"\n{'='*80}")
    print(f"📊 DIAGNOSTIC SUMMARY: {correct_count}/{num_samples} correct ({correct_count/num_samples*100:.1f}%)")
    print(f"{'='*80}")

    return correct_count, num_samples


def compare_models(client, data, num_samples=3):
    """Compare GPT-OSS-120B with other models on same questions."""

    models = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b"
    ]

    print(f"\n{'='*80}")
    print(f"🔄 SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")

    for i, record in enumerate(data[:num_samples]):
        print(f"\n{'━'*80}")
        print(f"📋 QUESTION {i+1}: {record['query']}")
        print(f"📚 EXPECTED: {record['answer']}")
        print(f"{'━'*80}")

        docs = record.get("positive", [])[:5]
        docs_text = "\n".join([f"Document {j+1}: {doc}" for j, doc in enumerate(docs)])

        user_prompt = f"""External Documents:
{docs_text}

Question: {record['query']}

Please answer the question based on the provided documents."""

        for model in models:
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

            is_correct = check_accuracy(llm_response, record['answer'])
            status = "✓" if is_correct else "✗"

            print(f"\n🤖 {model}")
            print(f"   Status: {status}")
            print(f"   Response: {llm_response[:200]}...")


def main():
    print("="*80)
    print("🔬 GPT-OSS-120B DIAGNOSTIC TOOL")
    print("="*80)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    refine_data = load_jsonl(DATA_DIR / "en_refine.json")

    print(f"\n📂 Loaded {len(refine_data)} records from en_refine.json")

    # Option 1: Diagnose GPT-OSS-120B in detail
    print("\n" + "="*80)
    print("PART 1: DETAILED GPT-OSS-120B ANALYSIS")
    print("="*80)
    diagnose_model(client, "openai/gpt-oss-120b", refine_data, num_samples=5)

    # Option 2: Side-by-side comparison
    print("\n" + "="*80)
    print("PART 2: SIDE-BY-SIDE MODEL COMPARISON")
    print("="*80)
    compare_models(client, refine_data, num_samples=3)


if __name__ == "__main__":
    main()