"""
Test script to find the maximum prompt length before Groq returns 413 error.
Uses binary search to efficiently find the threshold.

Usage:
    python test_groq_max_length.py
    python test_groq_max_length.py --model llama-3.3-70b-versatile
    python test_groq_max_length.py --model qwen/qwen3-32b --start 5000 --end 50000
"""

import os
import time
import argparse
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Models to test
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "moonshotai/kimi-k2-instruct-0905"
]


def get_api_key() -> str:
    for i in range(1, 21):
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key:
            return key
    key = os.getenv('GROQ_API_KEY')
    if key:
        return key
    raise ValueError("No GROQ_API_KEY found in environment")


def generate_text(length: int) -> str:
    """Generate dummy text of approximately the specified length."""
    base = "This is a test sentence for measuring API limits. "
    repeats = (length // len(base)) + 1
    return (base * repeats)[:length]


def test_length(client: Groq, model: str, context_len: int, question_len: int = 100) -> dict:
    """Test if a given length works. Returns dict with status and details."""
    context = generate_text(context_len)
    question = generate_text(question_len)

    prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

    total_len = len(prompt)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        return {
            "success": True,
            "context_len": context_len,
            "total_len": total_len,
            "response_len": len(response.choices[0].message.content)
        }
    except Exception as e:
        error_str = str(e)
        if "413" in error_str or "too large" in error_str.lower() or "request_too_large" in error_str:
            return {
                "success": False,
                "error_type": "413_too_large",
                "context_len": context_len,
                "total_len": total_len
            }
        elif "rate_limit" in error_str.lower() or "429" in error_str:
            return {
                "success": False,
                "error_type": "rate_limit",
                "context_len": context_len,
                "total_len": total_len
            }
        else:
            return {
                "success": False,
                "error_type": "other",
                "error": error_str[:200],
                "context_len": context_len,
                "total_len": total_len
            }


def binary_search_max_length(client: Groq, model: str, low: int, high: int, delay: float = 2.0) -> dict:
    """Binary search to find max working length."""
    print(f"\n🔍 Binary search for max length: {low} - {high} chars")
    print("-" * 50)

    last_success = low
    last_fail = high

    while high - low > 500:  # Stop when range is small enough
        mid = (low + high) // 2
        print(f"  Testing {mid:,} chars...", end=" ", flush=True)

        time.sleep(delay)  # Avoid rate limits
        result = test_length(client, model, mid)

        if result["success"]:
            print(f"✅ OK (total: {result['total_len']:,})")
            last_success = mid
            low = mid
        elif result["error_type"] == "413_too_large":
            print(f"❌ 413 Too Large (total: {result['total_len']:,})")
            last_fail = mid
            high = mid
        elif result["error_type"] == "rate_limit":
            print(f"⏳ Rate limited - waiting 30s...")
            time.sleep(30)
            continue  # Retry same length
        else:
            print(f"⚠️ Other error: {result.get('error', 'unknown')[:50]}")
            high = mid  # Treat as failure

    return {
        "max_success": last_success,
        "min_fail": last_fail,
        "threshold_range": f"{last_success:,} - {last_fail:,}"
    }


def quick_test(client: Groq, model: str, lengths: list, delay: float = 2.0) -> dict:
    """Quick test specific lengths to get a rough idea."""
    print(f"\n⚡ Quick test at specific lengths")
    print("-" * 50)

    results = {}
    for length in lengths:
        print(f"  Testing {length:,} chars...", end=" ", flush=True)
        time.sleep(delay)

        result = test_length(client, model, length)
        results[length] = result

        if result["success"]:
            print(f"✅ OK")
        elif result["error_type"] == "413_too_large":
            print(f"❌ 413 Too Large")
        elif result["error_type"] == "rate_limit":
            print(f"⏳ Rate limited")
        else:
            print(f"⚠️ Error: {result.get('error', 'unknown')[:30]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test Groq API max prompt length")
    parser.add_argument("--model", type=str, default="groq/compound", help="Model to test")
    parser.add_argument("--start", type=int, default=1000, help="Starting length for binary search")
    parser.add_argument("--end", type=int, default=30000, help="Ending length for binary search")
    parser.add_argument("--quick", action="store_true", help="Quick test at predefined lengths")
    parser.add_argument("--all-models", action="store_true", help="Test all models")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 GROQ API MAX LENGTH TESTER")
    print("=" * 60)

    api_key = get_api_key()
    client = Groq(api_key=api_key)
    print(f"✅ API initialized")

    models_to_test = MODELS if args.all_models else [args.model]

    all_results = {}

    for model in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"📦 MODEL: {model}")
        print("=" * 60)

        if args.quick:
            # Quick test at common lengths
            lengths = [25000,30000]
            results = quick_test(client, model, lengths, args.delay)

            # Find threshold from quick test
            success_lengths = [l for l, r in results.items() if r["success"]]
            fail_lengths = [l for l, r in results.items() if r.get("error_type") == "413_too_large"]

            if success_lengths and fail_lengths:
                max_success = max(success_lengths)
                min_fail = min(fail_lengths)
                print(f"\n📊 Quick Result: Works up to {max_success:,}, fails at {min_fail:,}")
                all_results[model] = {"max_success": max_success, "min_fail": min_fail}
            elif success_lengths:
                print(f"\n📊 Quick Result: All tested lengths worked (max tested: {max(success_lengths):,})")
                all_results[model] = {"max_success": max(success_lengths), "min_fail": None}
            else:
                print(f"\n📊 Quick Result: All tested lengths failed")
                all_results[model] = {"max_success": None, "min_fail": min(lengths)}
        else:
            # Full binary search
            result = binary_search_max_length(client, model, args.start, args.end, args.delay)
            all_results[model] = result

            print(f"\n📊 Result for {model}:")
            print(f"   Max working length: ~{result['max_success']:,} chars")
            print(f"   Threshold range: {result['threshold_range']}")

    # Final summary
    if len(models_to_test) > 1:
        print(f"\n{'=' * 60}")
        print("📊 SUMMARY - ALL MODELS")
        print("=" * 60)
        print(f"{'Model':<40} {'Max Working':<15} {'Min Fail':<15}")
        print("-" * 70)
        for model, result in all_results.items():
            max_s = f"{result.get('max_success', 'N/A'):,}" if result.get('max_success') else "N/A"
            min_f = f"{result.get('min_fail', 'N/A'):,}" if result.get('min_fail') else "N/A"
            print(f"{model:<40} {max_s:<15} {min_f:<15}")

    print(
        f"\n💡 Recommendation: Use context length < {min(r.get('max_success', 99999) for r in all_results.values() if r.get('max_success')):,} chars to be safe")


if __name__ == "__main__":
    main()