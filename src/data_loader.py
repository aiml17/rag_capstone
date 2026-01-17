"""
RAGBench Dataset Loader and Explorer
Focused on FinQA (Financial) domain for initial development.
"""

from datasets import load_dataset, get_dataset_config_names
import pandas as pd
from typing import Optional, List, Any


class RAGBenchLoader:
    """
    A class to load and explore the RAGBench dataset.
    Currently focused on FinQA (Financial) domain.
    """

    DATASET_NAME = "rungalileo/ragbench"
    FINANCE_DOMAIN = "finqa"

    def __init__(self):
        self.domains = None
        self.dataset = None

    def get_available_domains(self) -> List[str]:
        """Fetch all available domain configurations from the dataset."""
        print("Fetching available domains from RAGBench...")
        self.domains = get_dataset_config_names(self.DATASET_NAME)
        print(f"Found {len(self.domains)} domains: {self.domains}")
        return self.domains

    def load_finqa(self, split: str = "test") -> Any:
        """
        Load the FinQA (Financial) domain dataset.

        Args:
            split: Dataset split - 'train' or 'test'

        Returns:
            HuggingFace Dataset object
        """
        print(f"\nLoading '{self.FINANCE_DOMAIN}' dataset ({split} split)...")
        self.dataset = load_dataset(self.DATASET_NAME, self.FINANCE_DOMAIN, split=split)
        print(f"✅ Loaded {len(self.dataset)} examples from '{self.FINANCE_DOMAIN}'")
        return self.dataset

    def explore_dataset(self, num_samples: int = 1) -> None:
        """
        Print detailed information about the loaded dataset.

        Args:
            num_samples: Number of sample rows to display
        """
        if self.dataset is None:
            print("No dataset loaded. Call load_finqa() first.")
            return

        print("\n" + "=" * 80)
        print(f"DATASET STRUCTURE: {self.FINANCE_DOMAIN}")
        print("=" * 80)

        # Column names
        print(f"\nNumber of examples: {len(self.dataset)}")
        print(f"\nColumns ({len(self.dataset.column_names)}):")
        for col in self.dataset.column_names:
            print(f"  - {col}")

        # Sample data
        print(f"\n{'=' * 80}")
        print(f"SAMPLE DATA (first {num_samples} example)")
        print("=" * 80)

        for i in range(min(num_samples, len(self.dataset))):
            print(f"\n--- Example {i + 1} ---")
            example = self.dataset[i]
            for key, value in example.items():
                # Truncate long values for readability
                if isinstance(value, str) and len(value) > 300:
                    display_value = value[:300] + "... [truncated]"
                elif isinstance(value, list) and len(value) > 3:
                    display_value = f"{value[:2]}... [{len(value)} items total]"
                else:
                    display_value = value
                print(f"\n{key}:\n  {display_value}")

    def get_evaluation_columns(self) -> List[str]:
        """
        Identify columns related to RAGBench evaluation metrics.

        These are the ground truth scores we'll compare against:
        - context_relevance
        - context_utilization
        - completeness
        - adherence
        """
        if self.dataset is None:
            print("No dataset loaded. Call load_finqa() first.")
            return []

        eval_keywords = ['relevance', 'utilization', 'completeness',
                         'adherence', 'supported', 'score', 'faithfulness']
        eval_cols = [col for col in self.dataset.column_names
                     if any(kw in col.lower() for kw in eval_keywords)]
        return eval_cols

    def to_dataframe(self, num_rows: Optional[int] = None) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame for easier analysis."""
        if self.dataset is None:
            print("No dataset loaded. Call load_finqa() first.")
            return pd.DataFrame()

        if num_rows:
            return pd.DataFrame(self.dataset[:num_rows])
        return pd.DataFrame(self.dataset[:])


def main():
    """Main function to explore the FinQA dataset."""

    # Initialize loader
    loader = RAGBenchLoader()

    # Show all available domains
    print("=" * 80)
    print("STEP 1: FETCHING AVAILABLE DOMAINS")
    print("=" * 80)
    loader.get_available_domains()

    # Load FinQA dataset
    print("\n" + "=" * 80)
    print("STEP 2: LOADING FINQA DATASET")
    print("=" * 80)
    loader.load_finqa(split="test")

    # Explore the dataset structure
    print("\n" + "=" * 80)
    print("STEP 3: EXPLORING DATASET STRUCTURE")
    print("=" * 80)
    loader.explore_dataset(num_samples=1)

    # Show evaluation-related columns
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATION COLUMNS (Ground Truth Scores)")
    print("=" * 80)
    eval_cols = loader.get_evaluation_columns()
    print(f"Found {len(eval_cols)} evaluation columns:")
    for col in eval_cols:
        print(f"  - {col}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Domain: {loader.FINANCE_DOMAIN}")
    print(f"Number of examples: {len(loader.dataset)}")
    print(f"Number of columns: {len(loader.dataset.column_names)}")
    print(f"Evaluation columns: {eval_cols}")


if __name__ == "__main__":
    main()