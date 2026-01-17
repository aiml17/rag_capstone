"""
RGB Dataset Explorer
Run this to understand the structure of the dataset files.
"""

import json
from pathlib import Path

DATA_DIR = Path("data/rgb")

def load_jsonl(filepath: Path) -> list:
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def explore_file(filename: str):
    """Explore a single JSONL file."""
    filepath = DATA_DIR / filename

    print(f"\n{'='*60}")
    print(f"📁 FILE: {filename}")
    print('='*60)

    data = load_jsonl(filepath)

    print(f"\n📊 Format: JSONL (one JSON object per line)")
    print(f"📊 Total records: {len(data)}")

    # Get first record
    if len(data) > 0:
        first = data[0]
        print(f"\n🔑 Keys in first record:")
        for key in first.keys():
            value = first[key]
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"   • {key}: (str) {preview}")
            elif isinstance(value, list):
                print(f"   • {key}: (list) length={len(value)}")
                if len(value) > 0:
                    item = value[0]
                    if isinstance(item, str):
                        preview = item[:80] + "..." if len(item) > 80 else item
                        print(f"       └─ First item: {preview}")
                    elif isinstance(item, dict):
                        print(f"       └─ First item keys: {list(item.keys())}")
            elif isinstance(value, dict):
                print(f"   • {key}: (dict) keys={list(value.keys())}")
            else:
                print(f"   • {key}: ({type(value).__name__}) {value}")

        # Show full first record
        print(f"\n📝 First record (full):")
        print("-"*40)
        print(json.dumps(first, indent=2, ensure_ascii=False)[:2000])
        if len(json.dumps(first)) > 2000:
            print("... [truncated]")

        # Show second record for comparison
        if len(data) > 1:
            print(f"\n📝 Second record (full):")
            print("-"*40)
            print(json.dumps(data[1], indent=2, ensure_ascii=False)[:1500])
            if len(json.dumps(data[1])) > 1500:
                print("... [truncated]")


def main():
    print("🔍 RGB Dataset Explorer")
    print("="*60)

    files = ["en_refine.json", "en_int.json", "en_fact.json"]

    for filename in files:
        try:
            explore_file(filename)
        except FileNotFoundError:
            print(f"\n❌ File not found: {filename}")
        except Exception as e:
            print(f"\n❌ Error reading {filename}: {e}")

    print("\n" + "="*60)
    print("✅ Exploration complete!")


if __name__ == "__main__":
    main()