"""
Find checkpoint files with all-zero computed metrics.

Usage:
    python find_zero_checkpoints.py                    # Check all domains
    python find_zero_checkpoints.py --domain finqa    # Check specific domain
    python find_zero_checkpoints.py --delete          # Delete failed checkpoints (with confirmation)
"""

import os
import json
import glob
import argparse

BASE_DIR = "preliminary_evaluations"
DOMAINS = ["finqa", "cuad", "delucionqa", "hotpotqa", "covidqa"]


def check_checkpoint(filepath: str) -> dict:
    """Check a single checkpoint file for zero metrics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            return {"status": "empty", "total": 0, "zeros": 0, "valid": 0}

        total = len(results)

        zeros = 0
        valid = 0

        for r in results:
            is_valid = r.get("is_valid", False)
            if is_valid:
                valid += 1

            # Check if ALL computed metrics are zero
            rel = r.get("computed_relevance", 0)
            util = r.get("computed_utilization", 0)
            comp = r.get("computed_completeness", 0)
            adh = r.get("computed_adherence", 0)

            if rel == 0 and util == 0 and comp == 0 and adh == 0:
                zeros += 1

        if 10 != total:
            return {"status": "less_than_10", "total": total, "zeros": zeros, "valid": valid}

        if zeros == total and total > 0:
            return {"status": "all_zeros", "total": total, "zeros": zeros, "valid": valid}
        elif zeros > 0:
            return {"status": "partial_zeros", "total": total, "zeros": zeros, "valid": valid}
        else:
            return {"status": "ok", "total": total, "zeros": zeros, "valid": valid}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def scan_domain(domain: str) -> list:
    """Scan all checkpoints for a domain."""
    checkpoint_dir = os.path.join(BASE_DIR, domain, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        return []

    failed = []
    partial = []
    less_than_10 = []
    ok_count = 0

    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.json")))

    for filepath in files:
        filename = os.path.basename(filepath)
        result = check_checkpoint(filepath)

        if result["status"] == "all_zeros":
            failed.append({
                "file": filename,
                "path": filepath,
                "total": result["total"],
                "zeros": result["zeros"]
            })
        elif result["status"] == "partial_zeros":
            partial.append({
                "file": filename,
                "path": filepath,
                "total": result["total"],
                "zeros": result["zeros"],
                "valid": result["valid"]
            })
        elif result["status"] == "less_than_10":
            less_than_10.append({
                "file": filename,
                "path": filepath,
                "total": result["total"],
                "zeros": result["zeros"],
                "valid": result["valid"]
            })

        elif result["status"] == "ok":
            ok_count += 1

    return {
        "domain": domain,
        "total_files": len(files),
        "ok_count": ok_count,
        "failed": failed,
        "partial": partial,
        "less_than_10": less_than_10
    }


def main():
    parser = argparse.ArgumentParser(description="Find checkpoints with zero metrics")
    parser.add_argument("--domain", type=str, choices=DOMAINS, help="Check specific domain")
    parser.add_argument("--delete", action="store_true", help="Delete failed checkpoints")
    args = parser.parse_args()

    domains_to_check = [args.domain] if args.domain else DOMAINS

    all_failed = []
    all_partial = []

    print("\n" + "=" * 70)
    print("🔍 CHECKPOINT ZERO-VALUE SCANNER")
    print("=" * 70)

    for domain in domains_to_check:
        result = scan_domain(domain)

        if not result:
            print(f"\n⚠️  {domain.upper()}: No checkpoint directory found")
            continue

        print(f"\n📁 {domain.upper()}")
        print(f"   Total files: {result['total_files']}")
        print(f"   OK: {result['ok_count']}")
        print(f"   All zeros: {len(result['failed'])}")
        print(f"   Partial zeros: {len(result['partial'])}")
        print(f"   Less Count: {len(result['less_than_10'])}")

        if result['failed']:
            print(f"\n   ❌ ALL-ZERO checkpoints (completely failed):")
            for f in result['failed']:
                print(f"      - {f['file']} ({f['total']} examples, all zeros)")
                all_failed.append(f)

        if result['partial']:
            print(f"\n   ⚠️  PARTIAL-ZERO checkpoints (some failed):")
            for f in result['partial'][:10]:
                print(f"      - {f['file']} ({f['zeros']}/{f['total']} zeros, {f['valid']} valid)")
            if len(result['partial']) > 10:
                print(f"      ... and {len(result['partial']) - 10} more")
            all_partial.extend(result['partial'])

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"   Total ALL-ZERO checkpoints: {len(all_failed)}")
    print(f"   Total PARTIAL-ZERO checkpoints: {len(all_partial)}")

    if not all_failed:
        print("\n✅ No completely failed checkpoints found!")
        return

    # Delete option
    if args.delete and all_failed:
        print(f"\n⚠️  About to delete {len(all_failed)} checkpoint files:")
        for f in all_failed[:10]:
            print(f"   - {f['path']}")
        if len(all_failed) > 10:
            print(f"   ... and {len(all_failed) - 10} more")

        confirm = input(f"\nDelete these {len(all_failed)} files? [y/N]: ").strip().lower()

        if confirm == 'y':
            deleted = 0
            for f in all_failed:
                try:
                    os.remove(f['path'])
                    deleted += 1
                    print(f"   🗑️  Deleted: {f['file']}")
                except Exception as e:
                    print(f"   ⚠️  Error deleting {f['file']}: {e}")

            print(f"\n✅ Deleted {deleted}/{len(all_failed)} files")
            print("\n📋 Re-run evaluation to process these configs:")
            for domain in domains_to_check:
                print(f"   python run_preliminary_evaluation.py --domain {domain} --run")
        else:
            print("❌ Cancelled. No files deleted.")

    elif all_failed and not args.delete:
        print(f"\n💡 To delete failed checkpoints, run:")
        if args.domain:
            print(f"   python find_zero_checkpoints.py --domain {args.domain} --delete")
        else:
            print(f"   python find_zero_checkpoints.py --delete")


if __name__ == "__main__":
    main()