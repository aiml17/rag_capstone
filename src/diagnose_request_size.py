"""
Diagnose actual request sizes in RAGBench evaluation.
Checks what's really being sent to the API.

Usage:
    python diagnose_request_size.py --domain finqa
    python diagnose_request_size.py --domain cuad --samples 20
"""

import os
import sys
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.multidomain_loader import MultiDomainLoader
from src.chunker import DocumentChunker

DOMAINS = ["finqa", "cuad", "delucionqa", "hotpotqa", "covidqa"]
RANDOM_SEED = 42


def analyze_domain(domain: str, num_samples: int = 30):
    """Analyze request sizes for a domain."""
    print(f"\n{'=' * 70}")
    print(f"📊 ANALYZING: {domain.upper()}")
    print("=" * 70)

    # Load data
    loader = MultiDomainLoader()
    chunker = DocumentChunker()

    data = loader.load_domain(domain, "test")
    print(f"Total examples: {len(data)}")

    # Sample same as evaluation script
    random.seed(RANDOM_SEED)
    indices = list(range(len(data)))
    random.shuffle(indices)
    samples = [data[i] for i in indices[:num_samples]]

    # Track sizes
    stats = {
        "question_lens": [],
        "corpus_lens": [],
        "none_chunk_lens": [],
        "sentence_chunk_lens": [],
        "semantic_chunk_lens": [],
        "retrieved_5_lens": [],  # Sum of top 5 chunks
    }

    large_examples = []

    for i, sample in enumerate(samples):
        question = sample.get('question', '')
        documents = sample.get('documents', sample.get('context', []))

        if isinstance(documents, str):
            documents = [documents]
        corpus = "\n\n".join(documents)

        stats["question_lens"].append(len(question))
        stats["corpus_lens"].append(len(corpus))

        # Simulate chunking strategies
        # None chunking - entire corpus as one chunk
        none_chunks = [corpus] if corpus else ["No content"]
        stats["none_chunk_lens"].append(len(none_chunks[0]))

        # Sentence chunking
        sentence_results = chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
        sentence_chunks = [c['text'] for c in sentence_results] if sentence_results else [corpus]

        # Semantic chunking
        semantic_results = chunker.semantic_chunking(corpus)
        semantic_chunks = [c['text'] for c in semantic_results] if semantic_results else [corpus]

        # Simulate top-5 retrieval (worst case: largest 5 chunks)
        sentence_top5 = sorted(sentence_chunks, key=len, reverse=True)[:5]
        semantic_top5 = sorted(semantic_chunks, key=len, reverse=True)[:5]

        stats["sentence_chunk_lens"].append(sum(len(c) for c in sentence_top5))
        stats["semantic_chunk_lens"].append(sum(len(c) for c in semantic_top5))

        # For "none" chunking, retrieved = entire corpus
        stats["retrieved_5_lens"].append(len(corpus))

        # Track examples that would hit 413
        # Estimate full prompt size
        prompt_overhead = 100  # "Answer the question..." template

        for strategy, context_len in [
            ("none", len(corpus)),
            ("sentence", sum(len(c) for c in sentence_top5)),
            ("semantic", sum(len(c) for c in semantic_top5))
        ]:
            total = prompt_overhead + context_len + len(question)
            if total > 20000:  # Conservative threshold
                large_examples.append({
                    "idx": i,
                    "strategy": strategy,
                    "question_len": len(question),
                    "context_len": context_len,
                    "total_len": total,
                    "corpus_len": len(corpus)
                })

    # Print statistics
    print(f"\n📏 SIZE STATISTICS (from {num_samples} samples)")
    print("-" * 50)

    print(f"\n  Question lengths:")
    print(f"    Min: {min(stats['question_lens']):,}")
    print(f"    Max: {max(stats['question_lens']):,}")
    print(f"    Avg: {np.mean(stats['question_lens']):,.0f}")

    print(f"\n  Full corpus lengths:")
    print(f"    Min: {min(stats['corpus_lens']):,}")
    print(f"    Max: {max(stats['corpus_lens']):,}")
    print(f"    Avg: {np.mean(stats['corpus_lens']):,.0f}")

    print(f"\n  Context sizes by chunking strategy (top-5 retrieved):")
    print(
        f"    None (full corpus):  Avg={np.mean(stats['none_chunk_lens']):,.0f}, Max={max(stats['none_chunk_lens']):,}")
    print(
        f"    Sentence (top-5):    Avg={np.mean(stats['sentence_chunk_lens']):,.0f}, Max={max(stats['sentence_chunk_lens']):,}")
    print(
        f"    Semantic (top-5):    Avg={np.mean(stats['semantic_chunk_lens']):,.0f}, Max={max(stats['semantic_chunk_lens']):,}")

    # Estimated prompt sizes
    print(f"\n📦 ESTIMATED FULL PROMPT SIZES")
    print("-" * 50)

    prompt_overhead = 100
    avg_question = np.mean(stats['question_lens'])

    for strategy, lens in [
        ("none", stats['none_chunk_lens']),
        ("sentence", stats['sentence_chunk_lens']),
        ("semantic", stats['semantic_chunk_lens'])
    ]:
        total_lens = [l + q + prompt_overhead for l, q in zip(lens, stats['question_lens'])]
        print(f"\n  {strategy.upper()} chunking:")
        print(f"    Min: {min(total_lens):,}")
        print(f"    Max: {max(total_lens):,}")
        print(f"    Avg: {np.mean(total_lens):,.0f}")

        over_4k = sum(1 for t in total_lens if t > 4000)
        over_10k = sum(1 for t in total_lens if t > 10000)
        over_20k = sum(1 for t in total_lens if t > 20000)
        print(f"    >4k: {over_4k}/{num_samples} ({100 * over_4k / num_samples:.0f}%)")
        print(f"    >10k: {over_10k}/{num_samples} ({100 * over_10k / num_samples:.0f}%)")
        print(f"    >20k: {over_20k}/{num_samples} ({100 * over_20k / num_samples:.0f}%)")

    # Show problematic examples
    if large_examples:
        print(f"\n⚠️  EXAMPLES LIKELY TO HIT 413 (>20k chars)")
        print("-" * 50)

        # Group by strategy
        by_strategy = {}
        for ex in large_examples:
            s = ex['strategy']
            if s not in by_strategy:
                by_strategy[s] = []
            by_strategy[s].append(ex)

        for strategy, examples in by_strategy.items():
            print(f"\n  {strategy.upper()}: {len(examples)} problematic examples")
            for ex in examples[:3]:
                print(
                    f"    - Sample #{ex['idx']}: question={ex['question_len']:,}, context={ex['context_len']:,}, total={ex['total_len']:,}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Diagnose request sizes")
    parser.add_argument("--domain", type=str, choices=DOMAINS, help="Domain to analyze")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples")
    parser.add_argument("--all", action="store_true", help="Analyze all domains")
    args = parser.parse_args()

    print("=" * 70)
    print("🔍 REQUEST SIZE DIAGNOSTIC")
    print("=" * 70)

    if args.all:
        domains = DOMAINS
    elif args.domain:
        domains = [args.domain]
    else:
        domains = DOMAINS

    for domain in domains:
        try:
            analyze_domain(domain, args.samples)
        except Exception as e:
            print(f"\n❌ Error analyzing {domain}: {e}")

    print(f"\n{'=' * 70}")
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    print("""
  1. The "none" chunking strategy sends the ENTIRE corpus as context.
     This is likely causing 413 errors for large documents.

  2. Solutions:
     a) Truncate context in _generate_response() to max 15000 chars
     b) Skip "none" chunking for domains with large documents
     c) Check context length AFTER retrieval, before LLM call

  3. Current MAX_CONTEXT_LENGTH=4000 is too conservative.
     Based on testing, you can safely use 15000-20000.
""")


if __name__ == "__main__":
    main()