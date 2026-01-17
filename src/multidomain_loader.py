"""
Multi-Domain RAGBench Data Loader
Supports all 5 RAGBench domains: Finance, Legal, Customer Support, General Knowledge, Biomedical
"""

from datasets import load_dataset
from typing import List, Dict, Optional
import shutil
import os
import warnings
warnings.filterwarnings("ignore")


class MultiDomainLoader:
    """Loader for all RAGBench domains."""

    # Available RAGBench subsets:
    # covidqa, cuad, delucionqa, emanual, expertqa, finqa,
    # hagrid, hotpotqa, msmarco, pubmedqa, tatqa, delucionqa

    DOMAIN_INFO = {
        "finqa": {
            "name": "Finance",
            "description": "Financial question answering from earnings reports and SEC filings",
            "dataset_name": "rungalileo/ragbench",
            "subset": "finqa"
        },
        "cuad": {
            "name": "Legal",
            "description": "Contract Understanding Atticus Dataset - legal clause extraction",
            "dataset_name": "rungalileo/ragbench",
            "subset": "cuad"
        },
        "delucionqa": {
            "name": "Customer Support",
            "description": "Technical customer support Q&A from IT domain",
            "dataset_name": "rungalileo/ragbench",
            "subset": "delucionqa"
        },
        "hotpotqa": {
            "name": "General Knowledge",
            "description": "Multi-hop reasoning questions from Wikipedia",
            "dataset_name": "rungalileo/ragbench",
            "subset": "hotpotqa"
        },
        "covidqa": {
            "name": "Biomedical",
            "description": "COVID-19 biomedical research questions",
            "dataset_name": "rungalileo/ragbench",
            "subset": "covidqa"
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            cache_dir: Optional directory for caching datasets
        """
        self.cache_dir = cache_dir
        self._cached_datasets = {}

    @classmethod
    def list_domains(cls) -> List[str]:
        """Return list of available domains."""
        return list(cls.DOMAIN_INFO.keys())

    @classmethod
    def get_domain_info(cls, domain: str) -> Dict:
        """Get information about a specific domain."""
        if domain not in cls.DOMAIN_INFO:
            raise ValueError(f"Unknown domain: {domain}. Available: {cls.list_domains()}")
        return cls.DOMAIN_INFO[domain]

    def load_domain(self, domain: str, split: str = "test") -> List[Dict]:
        """
        Load data for a specific domain.

        Args:
            domain: Domain name (finqa, cuad, delucionqa, hotpotqa, covidqa)
            split: Dataset split (train, test, validation)

        Returns:
            List of samples with standardized fields
        """
        if domain not in self.DOMAIN_INFO:
            raise ValueError(f"Unknown domain: {domain}. Available: {self.list_domains()}")

        cache_key = f"{domain}_{split}"
        if cache_key in self._cached_datasets:
            return self._cached_datasets[cache_key]

        info = self.DOMAIN_INFO[domain]
        print(f"Loading {info['name']} ({domain}) - {split} split...")

        try:
            # First try normal loading
            dataset = load_dataset(
                info["dataset_name"],
                info["subset"],
                split=split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Initial load failed, trying force redownload...")
            try:
                # Try forcing redownload to bypass cache issues
                dataset = load_dataset(
                    info["dataset_name"],
                    info["subset"],
                    split=split,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    download_mode="force_redownload"
                )
            except Exception as e2:
                print(f"Force redownload failed, trying to clear cache...")
                try:
                    # Clear the specific cache directory
                    cache_path = os.path.expanduser("~/.cache/huggingface/datasets/rungalileo___ragbench")
                    if os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                        print(f"Cleared cache at {cache_path}")

                    # Try one more time
                    dataset = load_dataset(
                        info["dataset_name"],
                        info["subset"],
                        split=split,
                        trust_remote_code=True
                    )
                except Exception as e3:
                    raise RuntimeError(f"Failed to load {domain} after all attempts: {e3}")

        # Standardize the samples
        samples = self._standardize_samples(dataset, domain)
        self._cached_datasets[cache_key] = samples

        print(f"✓ Loaded {len(samples)} samples from {domain}")
        return samples

    def _standardize_samples(self, dataset, domain: str) -> List[Dict]:
        """
        Standardize sample format across domains.
        All domains will have: id, question, documents, response, and ground truth scores.
        """
        samples = []

        for idx, item in enumerate(dataset):
            sample = {
                # Standard ID
                "id": item.get("id") or item.get("idx") or item.get("example_id") or f"{domain}_{idx}",

                # Question
                "question": item.get("question") or item.get("query") or "",

                # Documents/Context (handle both list and string)
                "documents": self._extract_documents(item),

                # Ground truth response
                "response": item.get("response") or item.get("answer") or item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else item.get("answers", ""),

                # Ground truth TRACe scores
                "context_relevance": float(item.get("context_relevance", item.get("relevance_score", 0.5))),
                "context_utilization": float(item.get("context_utilization", item.get("utilization_score", 0.5))),
                "completeness": float(item.get("completeness", item.get("completeness_score", 0.5))),
                "adherence": self._extract_adherence(item),

                # Keep original for reference
                "_original": dict(item),
                "_domain": domain
            }
            samples.append(sample)

        return samples

    def _extract_documents(self, item: Dict) -> List[str]:
        """Extract documents from various possible field names."""
        # Try different field names
        docs = item.get("documents") or item.get("context") or item.get("contexts") or item.get("retrieved_contexts") or []

        if isinstance(docs, str):
            return [docs]
        elif isinstance(docs, list):
            # Flatten if nested
            flat_docs = []
            for d in docs:
                if isinstance(d, str):
                    flat_docs.append(d)
                elif isinstance(d, dict):
                    flat_docs.append(d.get("text", d.get("content", str(d))))
                elif isinstance(d, list):
                    flat_docs.extend([str(x) for x in d])
            return flat_docs
        return []

    def _extract_adherence(self, item: Dict) -> float:
        """Extract adherence score, handling both numeric and boolean."""
        adh = item.get("adherence", item.get("adherence_score", item.get("faithfulness", None)))

        if adh is None:
            return 1.0  # Default to adherent
        elif isinstance(adh, bool):
            return 1.0 if adh else 0.0
        elif isinstance(adh, (int, float)):
            return float(adh)
        elif isinstance(adh, str):
            return 1.0 if adh.lower() in ("true", "yes", "1") else 0.0
        return 1.0

    def load_all_domains(self, split: str = "test") -> Dict[str, List[Dict]]:
        """Load all domains at once."""
        all_data = {}
        for domain in self.list_domains():
            try:
                all_data[domain] = self.load_domain(domain, split)
            except Exception as e:
                print(f"⚠ Failed to load {domain}: {e}")
        return all_data

    def get_domain_stats(self, domain: str, split: str = "test") -> Dict:
        """Get statistics for a domain."""
        samples = self.load_domain(domain, split)

        return {
            "domain": domain,
            "name": self.DOMAIN_INFO[domain]["name"],
            "num_samples": len(samples),
            "avg_doc_length": sum(len(" ".join(s["documents"])) for s in samples) / len(samples) if samples else 0,
            "avg_num_docs": sum(len(s["documents"]) for s in samples) / len(samples) if samples else 0,
            "has_relevance_scores": sum(1 for s in samples if s["context_relevance"] != 0.5) / len(samples) if samples else 0,
        }


# Convenience functions
def load_finqa(split: str = "test") -> List[Dict]:
    return MultiDomainLoader().load_domain("finqa", split)

def load_cuad(split: str = "test") -> List[Dict]:
    return MultiDomainLoader().load_domain("cuad", split)

def load_delucionqa(split: str = "test") -> List[Dict]:
    return MultiDomainLoader().load_domain("delucionqa", split)

def load_hotpotqa(split: str = "test") -> List[Dict]:
    return MultiDomainLoader().load_domain("hotpotqa", split)

def load_covidqa(split: str = "test") -> List[Dict]:
    return MultiDomainLoader().load_domain("covidqa", split)


if __name__ == "__main__":
    # Test loading all domains
    loader = MultiDomainLoader()

    print("\n" + "="*60)
    print("RAGBench Multi-Domain Loader Test")
    print("="*60)

    for domain in loader.list_domains():
        print(f"\n--- {domain} ---")
        info = loader.get_domain_info(domain)
        print(f"Name: {info['name']}")
        print(f"Description: {info['description']}")

        try:
            stats = loader.get_domain_stats(domain)
            print(f"Samples: {stats['num_samples']}")
            print(f"Avg docs per sample: {stats['avg_num_docs']:.1f}")
        except Exception as e:
            print(f"Error: {e}")