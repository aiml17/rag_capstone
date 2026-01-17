"""
Document Chunking Module
Implements sentence-based and semantic chunking strategies.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentChunker:
    """
    Document chunker with multiple strategies:
    1. Sentence-based chunking with configurable overlap
    2. Semantic chunking based on embedding similarity
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the chunker.

        Args:
            embedding_model: Model name for semantic chunking
            device: Device to use ('mps', 'cuda', 'cpu')
            model: Pre-loaded SentenceTransformer model (to avoid reloading)
        """
        self.embedding_model_name = embedding_model
        self.model = model  # Can be set externally to share model
        self.device = device

    def _load_model(self):
        """Lazy load the embedding model if not already provided."""
        if self.model is None:
            print(f"Loading embedding model for semantic chunking: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name, device=self.device)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|vs|etc)\.\s', r'\1<PERIOD> ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore periods in abbreviations
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]

        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def sentence_chunking(
        self,
        text: str,
        chunk_size: int = 5,
        overlap: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using sentence-based strategy with overlap.

        Args:
            text: Input text to chunk
            chunk_size: Number of sentences per chunk
            overlap: Number of overlapping sentences between chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        # If document is small, return as single chunk
        if len(sentences) <= chunk_size:
            return [{
                'chunk_id': 0,
                'text': ' '.join(sentences),
                'start_sentence': 0,
                'end_sentence': len(sentences) - 1,
                'num_sentences': len(sentences),
                'strategy': 'sentence',
                'chunk_size': chunk_size,
                'overlap': overlap
            }]

        chunks = []
        start = 0
        chunk_id = 0

        # Ensure we make progress: step size must be at least 1
        step = max(1, chunk_size - overlap)

        while start < len(sentences):
            end = min(start + chunk_size, len(sentences))
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)

            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_sentence': start,
                'end_sentence': end - 1,
                'num_sentences': len(chunk_sentences),
                'strategy': 'sentence',
                'chunk_size': chunk_size,
                'overlap': overlap
            })

            chunk_id += 1

            # Move to next position
            start += step

            # If we've reached the end, stop
            if end >= len(sentences):
                break

        return chunks

    def semantic_chunking(
        self,
        text: str,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 2,
        max_chunk_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using semantic similarity between consecutive sentences.
        Splits when similarity drops below threshold.

        Args:
            text: Input text to chunk
            similarity_threshold: Split when similarity < threshold (0.0 to 1.0)
            min_chunk_size: Minimum sentences per chunk
            max_chunk_size: Maximum sentences per chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        self._load_model()

        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        # Single sentence - return as is
        if len(sentences) == 1:
            return [{
                'chunk_id': 0,
                'text': sentences[0],
                'start_sentence': 0,
                'end_sentence': 0,
                'num_sentences': 1,
                'strategy': 'semantic',
                'similarity_threshold': similarity_threshold
            }]

        # Small document - return as single chunk
        if len(sentences) <= min_chunk_size:
            return [{
                'chunk_id': 0,
                'text': ' '.join(sentences),
                'start_sentence': 0,
                'end_sentence': len(sentences) - 1,
                'num_sentences': len(sentences),
                'strategy': 'semantic',
                'similarity_threshold': similarity_threshold
            }]

        # Compute embeddings for all sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(float(sim))

        # Find chunk boundaries
        chunks = []
        chunk_id = 0
        start = 0

        for i in range(len(similarities)):
            current_size = i - start + 2  # +2 because we're looking at gap after sentence i

            # Decide whether to split
            should_split = False

            if current_size >= max_chunk_size:
                should_split = True
            elif current_size >= min_chunk_size and similarities[i] < similarity_threshold:
                should_split = True

            if should_split:
                end = i + 1  # Include sentence at index i
                chunk_sentences = sentences[start:end]
                chunk_text = ' '.join(chunk_sentences)

                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_sentence': start,
                    'end_sentence': end - 1,
                    'num_sentences': len(chunk_sentences),
                    'strategy': 'semantic',
                    'similarity_threshold': similarity_threshold
                })

                chunk_id += 1
                start = end

        # Add remaining sentences as final chunk
        if start < len(sentences):
            chunk_sentences = sentences[start:]
            chunk_text = ' '.join(chunk_sentences)

            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_sentence': start,
                'end_sentence': len(sentences) - 1,
                'num_sentences': len(chunk_sentences),
                'strategy': 'semantic',
                'similarity_threshold': similarity_threshold
            })

        return chunks

    def chunk_documents(
        self,
        documents: List[str],
        strategy: str = 'sentence',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents using specified strategy.

        Args:
            documents: List of document texts
            strategy: 'sentence' or 'semantic'
            **kwargs: Strategy-specific parameters

        Returns:
            List of all chunks with document references
        """
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            if not isinstance(doc, str) or not doc.strip():
                continue

            if strategy == 'sentence':
                chunks = self.sentence_chunking(doc, **kwargs)
            elif strategy == 'semantic':
                chunks = self.semantic_chunking(doc, **kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'sentence' or 'semantic'")

            # Add document reference to each chunk
            for chunk in chunks:
                chunk['doc_idx'] = doc_idx
                chunk['global_chunk_id'] = len(all_chunks)
                all_chunks.append(chunk)

        return all_chunks


def compare_chunking_strategies(text: str, chunker: DocumentChunker) -> Dict[str, Any]:
    """
    Compare sentence and semantic chunking on the same text.

    Args:
        text: Input text
        chunker: DocumentChunker instance

    Returns:
        Comparison results
    """
    # Sentence chunking with 5 sentences, 2 overlap
    sentence_chunks = chunker.sentence_chunking(text, chunk_size=5, overlap=2)

    # Semantic chunking
    semantic_chunks = chunker.semantic_chunking(text, similarity_threshold=0.5)

    return {
        'original_text_length': len(text),
        'total_sentences': len(chunker.split_into_sentences(text)),
        'sentence_chunking': {
            'num_chunks': len(sentence_chunks),
            'chunks': sentence_chunks
        },
        'semantic_chunking': {
            'num_chunks': len(semantic_chunks),
            'chunks': semantic_chunks
        }
    }


def main():
    """Test the chunking strategies."""

    print("=" * 80)
    print("DOCUMENT CHUNKING TEST")
    print("=" * 80)

    # Sample financial document (similar to FinQA)
    sample_text = """
    The company reported total revenue of $5.2 billion for fiscal year 2023. 
    This represents a 12% increase compared to the previous year. 
    Operating expenses decreased by 3% to $2.1 billion. 
    The gross profit margin improved to 45% from 42% in the prior year.
    Net income reached $890 million, up from $720 million in 2022.
    The board of directors approved a quarterly dividend of $0.50 per share.
    This dividend will be paid on March 15, 2024 to shareholders of record.
    Capital expenditures for the year totaled $450 million.
    The company invested heavily in research and development.
    R&D spending increased by 25% to support new product initiatives.
    Cash and cash equivalents stood at $1.8 billion at year end.
    Total debt was reduced by $200 million during the fiscal year.
    The company repurchased 5 million shares at an average price of $45.
    Management expects revenue growth of 8-10% for the coming year.
    New product launches are planned for the second quarter.
    """

    # Initialize chunker
    chunker = DocumentChunker()

    # Test sentence splitting
    print("\n" + "=" * 80)
    print("STEP 1: SENTENCE SPLITTING")
    print("=" * 80)
    sentences = chunker.split_into_sentences(sample_text)
    print(f"Total sentences found: {len(sentences)}")
    for i, sent in enumerate(sentences[:5]):
        print(f"  [{i}] {sent[:60]}...")

    # Test sentence-based chunking
    print("\n" + "=" * 80)
    print("STEP 2: SENTENCE-BASED CHUNKING (chunk_size=5, overlap=2)")
    print("=" * 80)
    sentence_chunks = chunker.sentence_chunking(sample_text, chunk_size=5, overlap=2)
    print(f"Number of chunks: {len(sentence_chunks)}")
    for chunk in sentence_chunks:
        print(f"\n  Chunk {chunk['chunk_id']}:")
        print(f"    Sentences: {chunk['start_sentence']} to {chunk['end_sentence']}")
        print(f"    Text: {chunk['text'][:100]}...")

    # Test semantic chunking
    print("\n" + "=" * 80)
    print("STEP 3: SEMANTIC CHUNKING (threshold=0.5)")
    print("=" * 80)
    semantic_chunks = chunker.semantic_chunking(sample_text, similarity_threshold=0.5)
    print(f"Number of chunks: {len(semantic_chunks)}")
    for chunk in semantic_chunks:
        print(f"\n  Chunk {chunk['chunk_id']}:")
        print(f"    Sentences: {chunk['start_sentence']} to {chunk['end_sentence']}")
        print(f"    Avg internal similarity: {chunk.get('avg_internal_similarity', 'N/A')}")
        print(f"    Text: {chunk['text'][:100]}...")

    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Total sentences in document: {len(sentences)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")

    avg_sent_chunk_size = np.mean([c['num_sentences'] for c in sentence_chunks])
    avg_sem_chunk_size = np.mean([c['num_sentences'] for c in semantic_chunks])
    print(f"Avg sentences per chunk (sentence-based): {avg_sent_chunk_size:.1f}")
    print(f"Avg sentences per chunk (semantic): {avg_sem_chunk_size:.1f}")

    print("\n" + "=" * 80)
    print("CHUNKING TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()