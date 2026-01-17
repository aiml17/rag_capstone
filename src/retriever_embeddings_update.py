"""
Update this EMBEDDING_MODELS dict in your src/retriever.py file
to add domain-specific embedding models.
"""

# Replace your existing EMBEDDING_MODELS dict with this:

EMBEDDING_MODELS = {
    # ============================================
    # General Purpose Models (all domains)
    # ============================================
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",

    # ============================================
    # Finance Domain
    # ============================================
    "finbert": "ProsusAI/finbert",

    # ============================================
    # Legal Domain
    # ============================================
    "legal-bert": "nlpaueb/legal-bert-base-uncased",
    # "caselaw-bert": "casehold/legalbert",  # Alternative

    # ============================================
    # Biomedical Domain
    # ============================================
    "pubmedbert": "NeuML/pubmedbert-base-embeddings",
    "s-pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",
    "biobert": "gsarti/biobert-nli",
}

# ============================================
# Usage Example
# ============================================
"""
from src.retriever import UnifiedRetriever

# Finance
retriever = UnifiedRetriever(method="finbert")

# Legal
retriever = UnifiedRetriever(method="legal-bert")

# Biomedical
retriever = UnifiedRetriever(method="pubmedbert")

# General
retriever = UnifiedRetriever(method="bge-base")
"""

# ============================================
# Test Script - Run this to verify models load
# ============================================
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Testing domain-specific embedding models...")
    print("=" * 60)

    test_text = "This is a test sentence for embedding."

    for name, model_id in EMBEDDING_MODELS.items():
        try:
            print(f"\n{name}: {model_id}")
            model = SentenceTransformer(model_id)
            embedding = model.encode(test_text)
            print(f"  ✓ Loaded successfully, embedding dim: {len(embedding)}")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:50]}")

    print("\n" + "=" * 60)
    print("Done!")