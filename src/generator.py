"""
RAG Generator Module
Uses Groq API with Llama/Mistral models to generate responses.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()


class RAGGenerator:
    """
    Generator component for RAG pipeline.
    Uses Groq API for fast LLM inference.
    """

    # Available models on Groq (as of December 2024)
    MODELS = {
        'llama-3.3-70b': 'llama-3.3-70b-versatile',                    # Best quality, recommended
        'llama-3.1-8b': 'llama-3.1-8b-instant',                        # Fast, good quality
        'llama-4-scout': 'meta-llama/llama-4-scout-17b-16e-instruct',  # Llama 4 Scout
        'llama-4-maverick': 'meta-llama/llama-4-maverick-17b-128e-instruct',  # Llama 4 Maverick
        'qwen3-32b': 'qwen/qwen3-32b',                                 # Alibaba Qwen
        'kimi-k2': 'moonshotai/kimi-k2-instruct',                      # Moonshot AI
        'gpt-oss-20b': 'openai/gpt-oss-20b',                           # OpenAI open source
        'gpt-oss-120b': 'openai/gpt-oss-120b',                         # OpenAI large
        'compound': 'groq/compound',                                   # Groq compound
    }

    def __init__(
        self,
        model_name: str = 'llama-3.1-8b',
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """
        Initialize the generator.

        Args:
            model_name: Model to use ('llama-3.1-8b', 'llama-3.3-70b', 'llama-4-scout', 'qwen3-32b', etc.)
            api_key: Groq API key (uses env var GROQ_API_KEY if not provided)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GROQ_API_KEY')

        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Either pass it as parameter or "
                "set GROQ_API_KEY environment variable in .env file"
            )

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)

        # Set model
        if model_name in self.MODELS:
            self.model = self.MODELS[model_name]
        else:
            self.model = model_name  # Allow direct model string

        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"Initialized RAG Generator with model: {self.model}")

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of retrieved chunk dictionaries

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            score = chunk.get('score', 0)
            context_parts.append(f"[Document {i}] (relevance: {score:.2f})\n{text}")

        return "\n\n".join(context_parts)

    def _build_prompt(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build the prompt messages for the LLM.
        Uses the LONG + CoT prompt from "Best Practices in RAG" paper.

        Args:
            question: User's question
            context: Retrieved context
            system_prompt: Custom system prompt (optional)

        Returns:
            List of message dictionaries
        """
        if system_prompt is None:
            # LONG + CoT prompt from "Searching for Best Practices in RAG" paper
            system_prompt = """You are a chatbot providing answers to user queries. You will be given one or more context documents. Use the information in the documents to answer the question.

If the documents do not provide enough information for you to answer the question, then say "The documents are missing some of the information required to answer the question." Don't quote facts not in the documents. Don't try to make up an answer.

Think step by step and explain your reasoning, quoting the documents when necessary."""

        user_prompt = f"""Context Documents:
{context}

Question: {question}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def generate(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response for the given question using retrieved chunks.

        Args:
            question: User's question
            chunks: List of retrieved chunk dictionaries
            system_prompt: Custom system prompt (optional)

        Returns:
            Dictionary with response and metadata
        """
        # Build context from chunks
        context = self._build_context(chunks)

        # Build prompt messages
        messages = self._build_prompt(question, context, system_prompt)

        # Call Groq API
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_text = completion.choices[0].message.content

            return {
                'response': response_text,
                'model': self.model,
                'usage': {
                    'prompt_tokens': completion.usage.prompt_tokens,
                    'completion_tokens': completion.usage.completion_tokens,
                    'total_tokens': completion.usage.total_tokens
                },
                'context_used': context,
                'num_chunks': len(chunks)
            }

        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'model': self.model,
                'error': str(e),
                'context_used': context,
                'num_chunks': len(chunks)
            }

    def generate_batch(
        self,
        examples: List[Dict[str, Any]],
        chunk_key: str = 'retrieved_chunks',
        question_key: str = 'question'
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple examples.

        Args:
            examples: List of example dictionaries
            chunk_key: Key for retrieved chunks in each example
            question_key: Key for question in each example

        Returns:
            List of examples with added 'generated_response' field
        """
        results = []

        for example in examples:
            question = example.get(question_key, '')
            chunks = example.get(chunk_key, [])

            # Generate response
            generation = self.generate(question, chunks)

            # Add to example
            result = example.copy()
            result['generated_response'] = generation['response']
            result['generation_metadata'] = {
                'model': generation['model'],
                'usage': generation.get('usage', {}),
                'num_chunks_used': generation['num_chunks']
            }

            results.append(result)

        return results


def main():
    """Test the RAG Generator."""

    print("=" * 80)
    print("RAG GENERATOR TEST")
    print("=" * 80)

    # Initialize generator
    print("\nInitializing generator...")
    try:
        generator = RAGGenerator(model_name='llama-3.1-8b')
    except ValueError as e:
        print(f"Error: {e}")
        print("\nMake sure you have a .env file with GROQ_API_KEY=your_key")
        return

    # Test with sample data
    print("\n" + "=" * 80)
    print("TEST 1: SIMPLE FINANCIAL QUESTION")
    print("=" * 80)

    sample_chunks = [
        {
            'text': 'The company reported total revenue of $5.2 billion for fiscal year 2023, representing a 12% increase from the previous year.',
            'score': 0.85
        },
        {
            'text': 'Net income reached $890 million, up from $720 million in 2022, reflecting improved operational efficiency.',
            'score': 0.78
        },
        {
            'text': 'The board approved a quarterly dividend of $0.50 per share, payable on March 15, 2024.',
            'score': 0.65
        }
    ]

    question = "What was the company's revenue and how did it compare to last year?"

    print(f"\nQuestion: {question}")
    print(f"Number of context chunks: {len(sample_chunks)}")

    result = generator.generate(question, sample_chunks)

    print(f"\n{'─' * 40}")
    print("GENERATED RESPONSE:")
    print('─' * 40)
    print(result['response'])
    print(f"\n{'─' * 40}")
    print("METADATA:")
    print('─' * 40)
    print(f"  Model: {result['model']}")
    if 'usage' in result:
        print(f"  Tokens used: {result['usage']['total_tokens']}")

    # Test 2: Question with insufficient context
    print("\n" + "=" * 80)
    print("TEST 2: QUESTION WITH LIMITED CONTEXT")
    print("=" * 80)

    question2 = "What is the company's market share in Asia?"

    print(f"\nQuestion: {question2}")

    result2 = generator.generate(question2, sample_chunks)

    print(f"\n{'─' * 40}")
    print("GENERATED RESPONSE:")
    print('─' * 40)
    print(result2['response'])

    # Test 3: FinQA-style question
    print("\n" + "=" * 80)
    print("TEST 3: FINQA-STYLE CALCULATION QUESTION")
    print("=" * 80)

    finqa_chunks = [
        {
            'text': '[["", "2022", "2023"], ["Revenue", "4.64", "5.20"], ["Net Income", "0.72", "0.89"], ["EPS", "1.20", "1.48"]]',
            'score': 0.92
        },
        {
            'text': 'All figures are in billions of dollars except EPS which is in dollars per share.',
            'score': 0.75
        }
    ]

    question3 = "What was the percentage increase in net income from 2022 to 2023?"

    print(f"\nQuestion: {question3}")

    result3 = generator.generate(question3, finqa_chunks)

    print(f"\n{'─' * 40}")
    print("GENERATED RESPONSE:")
    print('─' * 40)
    print(result3['response'])

    print("\n" + "=" * 80)
    print("GENERATOR TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()