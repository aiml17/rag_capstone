"""
TRACe Metrics Evaluator for RAG Pipeline
Uses LLM-as-Judge approach from RAGBench paper (Appendix 7.4)
Supports: Groq API (cloud) or Ollama (local)
"""

import os
import sys
import json
import re
import time
import requests

# ============================================================================
# DISABLE ALL TELEMETRY
# ============================================================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Try to import Groq (optional if using local)
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


# ============================================================================
# RAGBench LLM-as-Judge Prompt (from Appendix 7.4)
# ============================================================================

TRACE_EVALUATION_PROMPT = '''I asked someone to answer a question based on one or more documents.
Your task is to review their response and assess whether or not each sentence
in that response is supported by text in the documents. And if so, which
sentences in the documents provide that support. You will also tell me which
of the documents contain useful information for answering the question, and
which of the documents the answer was sourced from.

Here are the documents, each of which is split into sentences. Alongside each
sentence is associated key, such as '0a.' or '0b.' that you can use to refer
to it:
\'\'\'
{documents}
\'\'\'

The question was:
\'\'\'
{question}
\'\'\'

Here is their response, split into sentences. Alongside each sentence is
associated key, such as 'a.' or 'b.' that you can use to refer to it. Note
that these keys are unique to the response, and are not related to the keys
in the documents:
\'\'\'
{answer}
\'\'\'

You must respond with a JSON object matching this schema:
\'\'\'
{{
    "relevance_explanation": string,
    "all_relevant_sentence_keys": [string],
    "overall_supported_explanation": string,
    "overall_supported": boolean,
    "sentence_support_information": [
        {{
            "response_sentence_key": string,
            "explanation": string,
            "supporting_sentence_keys": [string],
            "fully_supported": boolean
        }},
    ],
    "all_utilized_sentence_keys": [string]
}}
\'\'\'

The relevance_explanation field is a string explaining which documents
contain useful information for answering the question.

The all_relevant_sentence_keys field is a list of all document sentences keys
(e.g. '0a') that are relevant to the question.

The overall_supported_explanation field is a string explaining why the response
*as a whole* is or is not supported by the documents.

The overall_supported field is a boolean indicating whether the response as a
whole is supported by the documents.

The sentence_support_information field is a list of objects, one for each sentence
in the response with: response_sentence_key, explanation, supporting_sentence_keys, fully_supported.

The all_utilized_sentence_keys field is a list of all sentences keys (e.g. '0a') that
were used to construct the answer.

You must respond with a valid JSON string only. No markdown, no backticks.'''


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TRACeScores:
    """Container for TRACe metric scores."""
    context_relevance: float
    context_utilization: float
    completeness: float
    adherence: bool

    def adherence_score(self) -> float:
        return 1.0 if self.adherence else 0.0

    def average(self) -> float:
        return np.mean([
            self.context_relevance,
            self.context_utilization,
            self.completeness,
            self.adherence_score()
        ])


@dataclass
class LLMJudgeResponse:
    """Container for parsed LLM judge response."""
    relevance_explanation: str
    all_relevant_sentence_keys: List[str]
    overall_supported_explanation: str
    overall_supported: bool
    sentence_support_information: List[Dict]
    all_utilized_sentence_keys: List[str]
    raw_response: str = ""


@dataclass
class EvaluationResult:
    """Container for a single evaluation result."""
    question: str
    ground_truth: str
    generated_response: str
    retrieved_context: str
    trace_scores: TRACeScores
    llm_judge_response: Optional[LLMJudgeResponse] = None
    total_context_sentences: int = 0
    relevant_sentences: int = 0
    utilized_sentences: int = 0
    gt_relevance: Optional[float] = None
    gt_utilization: Optional[float] = None
    gt_completeness: Optional[float] = None
    gt_adherence: Optional[bool] = None


# ============================================================================
# TRACe Evaluator Class
# ============================================================================

class TRACeEvaluator:
    """
    LLM-as-Judge Evaluator for TRACe metrics.
    Supports both Groq API (cloud) and Ollama (local).
    """

    def __init__(
        self,
        judge_model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        verbose: bool = False,
        use_local: bool = False,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the evaluator.

        Args:
            judge_model: Model name (Groq model or Ollama model)
            api_key: Groq API key (not needed for local)
            max_retries: Number of retries
            retry_delay: Delay between retries
            verbose: Print debug info
            use_local: Use local Ollama instead of Groq API
            ollama_base_url: Ollama API URL
        """
        self.judge_model = judge_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verbose = verbose
        self.use_local = use_local
        self.ollama_base_url = ollama_base_url

        if use_local:
            # Test Ollama connection
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = [m['name'] for m in response.json().get('models', [])]
                    if judge_model not in models and f"{judge_model}:latest" not in models:
                        print(f"⚠️ Model '{judge_model}' not found in Ollama. Available: {models[:5]}")
                    print(f"TRACe Evaluator initialized with LOCAL model: {judge_model}")
                else:
                    raise ConnectionError("Ollama not responding")
            except Exception as e:
                raise ConnectionError(f"Cannot connect to Ollama at {ollama_base_url}: {e}\n"
                                     f"Make sure Ollama is running: ollama serve")
        else:
            # Use Groq API
            if not HAS_GROQ:
                raise ImportError("groq package not installed. Run: pip install groq")
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
            self.client = Groq(api_key=api_key)
            print(f"TRACe Evaluator initialized with judge model: {judge_model}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text or not text.strip():
            return []

        text = text.strip()
        abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
                        'Inc.', 'Ltd.', 'Corp.', 'vs.', 'etc.', 'e.g.', 'i.e.',
                        'U.S.', 'U.K.', 'E.U.', 'No.', 'Vol.', 'Fig.']

        protected_text = text
        for abbr in abbreviations:
            protected_text = protected_text.replace(abbr, abbr.replace('.', '<PERIOD>'))

        sentences = re.split(r'(?<=[.!?])\s+', protected_text)
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
        sentences = [s for s in sentences if s]

        return sentences if sentences else [text]

    def _format_documents_with_keys(self, documents: List[str]) -> Tuple[str, int]:
        """Format documents with sentence keys."""
        formatted_parts = []
        total_sentences = 0

        for doc_idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc)
            for sent_idx, sentence in enumerate(sentences):
                key = f"{doc_idx}{chr(ord('a') + sent_idx)}"
                formatted_parts.append(f"{key}. {sentence}")
                total_sentences += 1

        return "\n".join(formatted_parts), total_sentences

    def _format_response_with_keys(self, response: str) -> Tuple[str, int]:
        """Format response with sentence keys."""
        sentences = self._split_into_sentences(response)
        formatted_parts = []

        for idx, sentence in enumerate(sentences):
            key = chr(ord('a') + idx)
            formatted_parts.append(f"{key}. {sentence}")

        return "\n".join(formatted_parts), len(sentences)

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama model."""
        url = f"{self.ollama_base_url}/api/generate"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json={
                        "model": self.judge_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 4096
                        }
                    },
                    timeout=300  # 2 minute timeout for long responses
                )

                if response.status_code == 200:
                    return response.json().get('response', '')
                else:
                    if self.verbose:
                        print(f"\n    Ollama error: {response.status_code} - {response.text[:100]}")
                    time.sleep(self.retry_delay)

            except requests.exceptions.Timeout:
                if self.verbose:
                    print(f"\n    Ollama timeout, retry {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay)
            except Exception as e:
                if self.verbose:
                    print(f"\n    Ollama error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise e

        return ""

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()

                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    wait_time = self.retry_delay * (2 ** attempt)
                    if self.verbose:
                        print(f"\n    ⚠️ Rate limit. Waiting {wait_time:.0f}s (retry {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"\n    Retry {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise e
        return ""

    def _call_llm_judge(self, prompt: str) -> str:
        """Call the LLM judge (local or cloud)."""
        if self.use_local:
            return self._call_ollama(prompt)
        else:
            return self._call_groq(prompt)

    def _parse_llm_response(self, response_text: str) -> LLMJudgeResponse:
        """Parse the JSON response from the LLM judge."""
        if not response_text:
            return self._empty_response()

        cleaned = response_text.strip()

        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Try parsing strategies
        parsing_attempts = [
            ("direct", lambda x: x),
            ("extract_json", self._extract_json_object),
            ("fix_escapes", self._fix_json_escapes),
        ]

        for attempt_name, fix_func in parsing_attempts:
            try:
                fixed_json = fix_func(cleaned)
                data = json.loads(fixed_json)

                return LLMJudgeResponse(
                    relevance_explanation=data.get("relevance_explanation", ""),
                    all_relevant_sentence_keys=data.get("all_relevant_sentence_keys", []),
                    overall_supported_explanation=data.get("overall_supported_explanation", ""),
                    overall_supported=data.get("overall_supported", False),
                    sentence_support_information=data.get("sentence_support_information", []),
                    all_utilized_sentence_keys=data.get("all_utilized_sentence_keys", []),
                    raw_response=response_text
                )
            except (json.JSONDecodeError, Exception):
                continue

        # Fallback: manual extraction
        if self.verbose:
            print(f"    [Debug] Using fallback JSON extraction")
        return self._manual_field_extraction(cleaned, response_text)

    def _empty_response(self) -> LLMJudgeResponse:
        """Return empty response object."""
        return LLMJudgeResponse(
            relevance_explanation="",
            all_relevant_sentence_keys=[],
            overall_supported_explanation="",
            overall_supported=False,
            sentence_support_information=[],
            all_utilized_sentence_keys=[],
            raw_response=""
        )

    def _extract_json_object(self, text: str) -> str:
        """Extract JSON object from text."""
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text

    def _fix_json_escapes(self, text: str) -> str:
        """Fix common JSON escape issues."""
        text = self._extract_json_object(text)
        valid_escapes = ['n', 'r', 't', 'b', 'f', '"', '\\', '/', 'u']

        result = []
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                next_char = text[i + 1]
                if next_char in valid_escapes:
                    result.append(text[i:i+2])
                    i += 2
                elif next_char == "'":
                    result.append("'")
                    i += 2
                else:
                    result.append(next_char)
                    i += 2
            else:
                result.append(text[i])
                i += 1

        return ''.join(result)

    def _manual_field_extraction(self, text: str, raw_response: str) -> LLMJudgeResponse:
        """Manually extract key fields when JSON parsing fails."""
        overall_supported = False
        if '"overall_supported": true' in text.lower() or '"overall_supported":true' in text.lower():
            overall_supported = True

        relevant_keys = []
        utilized_keys = []

        relevant_match = re.search(r'"all_relevant_sentence_keys"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if relevant_match:
            keys_str = relevant_match.group(1)
            relevant_keys = re.findall(r'"(\d+[a-z])"', keys_str)

        utilized_match = re.search(r'"all_utilized_sentence_keys"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if utilized_match:
            keys_str = utilized_match.group(1)
            utilized_keys = re.findall(r'"(\d+[a-z])"', keys_str)

        return LLMJudgeResponse(
            relevance_explanation="(extracted via fallback)",
            all_relevant_sentence_keys=relevant_keys,
            overall_supported_explanation="(extracted via fallback)",
            overall_supported=overall_supported,
            sentence_support_information=[],
            all_utilized_sentence_keys=utilized_keys,
            raw_response=raw_response
        )

    def _compute_trace_metrics(
        self,
        judge_response: LLMJudgeResponse,
        total_context_sentences: int
    ) -> TRACeScores:
        """Compute TRACe metrics from LLM judge response."""
        relevant_keys = set(judge_response.all_relevant_sentence_keys)
        utilized_keys = set(judge_response.all_utilized_sentence_keys)

        if total_context_sentences == 0:
            return TRACeScores(
                context_relevance=0.0,
                context_utilization=0.0,
                completeness=0.0,
                adherence=judge_response.overall_supported
            )

        context_relevance = len(relevant_keys) / total_context_sentences
        context_utilization = len(utilized_keys) / total_context_sentences

        if len(relevant_keys) > 0:
            intersection = relevant_keys.intersection(utilized_keys)
            completeness = len(intersection) / len(relevant_keys)
        else:
            completeness = 0.0

        return TRACeScores(
            context_relevance=min(context_relevance, 1.0),
            context_utilization=min(context_utilization, 1.0),
            completeness=min(completeness, 1.0),
            adherence=judge_response.overall_supported
        )

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        generated_response: str,
        retrieved_chunks: List[str],
        gt_scores: Optional[Dict] = None
    ) -> EvaluationResult:
        """Evaluate a single RAG example using LLM-as-Judge."""
        formatted_docs, total_doc_sentences = self._format_documents_with_keys(retrieved_chunks)
        formatted_response, response_sentences = self._format_response_with_keys(generated_response)

        prompt = TRACE_EVALUATION_PROMPT.format(
            documents=formatted_docs,
            question=question,
            answer=formatted_response
        )

        llm_response = self._call_llm_judge(prompt)
        judge_response = self._parse_llm_response(llm_response)
        trace_scores = self._compute_trace_metrics(judge_response, total_doc_sentences)

        result = EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            generated_response=generated_response,
            retrieved_context=" ".join(retrieved_chunks[:2])[:500],
            trace_scores=trace_scores,
            llm_judge_response=judge_response,
            total_context_sentences=total_doc_sentences,
            relevant_sentences=len(judge_response.all_relevant_sentence_keys),
            utilized_sentences=len(judge_response.all_utilized_sentence_keys)
        )

        if gt_scores:
            result.gt_relevance = gt_scores.get('relevance_score')
            result.gt_utilization = gt_scores.get('utilization_score')
            result.gt_completeness = gt_scores.get('completeness_score')
            result.gt_adherence = gt_scores.get('adherence_score')

        return result


# ============================================================================
# Test Function
# ============================================================================

def main():
    """Test the evaluator with local and cloud options."""
    print("="*60)
    print("TRACe EVALUATOR TEST")
    print("="*60)

    # Test with local Ollama
    print("\n--- Testing LOCAL (Ollama) ---")
    try:
        evaluator = TRACeEvaluator(
            judge_model="qwen2.5:7b",  # Change to your installed model
            use_local=True,
            verbose=True
        )

        test_example = {
            'question': "What was the company's revenue?",
            'ground_truth': "Revenue was $5.2 billion.",
            'generated_response': "The company reported revenue of $5.2 billion in 2023.",
            'retrieved_chunks': [
                "The company reported total revenue of $5.2 billion for fiscal year 2023."
            ]
        }

        result = evaluator.evaluate_single(**test_example)
        print(f"\n✓ Local evaluation successful!")
        print(f"  TRACe Score: {result.trace_scores.average():.4f}")
        print(f"  Adherence: {result.trace_scores.adherence}")

    except Exception as e:
        print(f"✗ Local test failed: {e}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()