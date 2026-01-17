"""
RAG Pipeline Demo UI - Streamlit Version
Interactive interface to test RAG configurations across domains.

Usage:
    streamlit run rag_demo_streamlit.py

Requirements:
    pip install streamlit
"""

import os
import sys
import random
import uuid
import shutil
import glob
import json
import re
import time
from pathlib import Path

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try both import styles (works whether file is in src/ or project root)
try:
    from src.multidomain_loader import MultiDomainLoader
    from src.chunker import DocumentChunker
    from src.evaluator import TRACeEvaluator
    from src.retriever import UnifiedRetriever
    from src.hybrid_retriever import HybridRetriever
except ModuleNotFoundError:
    from multidomain_loader import MultiDomainLoader
    from chunker import DocumentChunker
    from evaluator import TRACeEvaluator
    from retriever import UnifiedRetriever
    from hybrid_retriever import HybridRetriever

from groq import Groq

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TOP_K = 5
NUM_QUESTIONS_PER_DOMAIN = 10
CACHE_DIR = Path("./dataset_cache")

JUDGE_MODEL = "qwen2.5:7b-instruct"

DOMAINS = ["finqa", "cuad", "delucionqa", "hotpotqa", "covidqa"]

DOMAIN_NAMES = {
    "finqa": "Finance (FinQA)",
    "cuad": "Legal (CUAD)",
    "delucionqa": "Customer Support (DelucionQA)",
    "hotpotqa": "General Knowledge (HotpotQA)",
    "covidqa": "Biomedical (CovidQA)"
}

CHUNKING_STRATEGIES = ["none", "sentence", "semantic"]
RETRIEVAL_TYPES = ["dense", "sparse", "hybrid"]

DOMAIN_EMBEDDINGS = {
    "finqa": ["finbert", "bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "cuad": ["legal-bert", "e5-large", "bge-large", "minilm", "mpnet"],
    "delucionqa": ["bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "hotpotqa": ["bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "covidqa": ["biobert", "sapbert", "pubmedbert", "bge-base", "minilm", "mpnet"]
}

SPARSE_METHODS = ["bm25", "tfidf"]

LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "groq/compound"
]

# ============================================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================================

st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Header */
    .main-header {
        color: #818cf8;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        padding-top: 10px;
    }
    .main-header span {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 0.95rem;
        margin: 4px 0 15px 0;
    }
    
    /* Timer and Dataset ID row */
    .id-timer-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .timer-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 2px solid #6366f1;
        border-radius: 8px;
        padding: 8px 16px;
        color: #22d3ee;
        font-family: monospace;
        font-size: 1rem;
        font-weight: 600;
    }
    .timer-box.completed {
        border-color: #22c55e;
        color: #22c55e;
    }
    
    /* Section titles */
    .section-title {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 10px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-card:hover {
        border-color: #6366f1;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: monospace;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 5px;
    }
    .metric-r { color: #34d399; }
    .metric-u { color: #60a5fa; }
    .metric-c { color: #fbbf24; }
    .metric-a { color: #f472b6; }
    
    /* Result boxes */
    .result-box {
        background: #0f172a;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #e2e8f0;
        line-height: 1.6;
    }
    .answer-box { border-color: #4f46e5; }
    .ground-truth-box { border-color: #22c55e; }
    .context-box { border-color: #475569; font-size: 0.85rem; }
    
    .box-label {
        color: #c7d2fe;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    
    /* Dataset ID */
    .dataset-id {
        background: linear-gradient(135deg, #312e81 0%, #4c1d95 100%);
        border: 2px solid #6366f1;
        border-radius: 8px;
        padding: 8px 16px;
        color: #e0e7ff;
        font-family: monospace;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    
    /* Selectboxes */
    .stSelectbox > div > div {
        background-color: #0f172a;
        border: 2px solid #475569;
        border-radius: 8px;
        color: #e2e8f0;
    }
    .stSelectbox > div > div:hover {
        border-color: #6366f1;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    
    /* Footer */
    .footer {
        color: #64748b;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        border-top: 1px solid #334155;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Labels */
    .stSelectbox label, .stTextArea label {
        color: #c7d2fe !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

def get_cache_path(domain: str) -> Path:
    return CACHE_DIR / f"{domain}_samples.json"

def save_samples_to_cache(domain: str, samples: list):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = get_cache_path(domain)
    serializable = []
    for s in samples:
        sample = dict(s)
        if '_original' in sample:
            try:
                sample['_original'] = dict(sample['_original'])
            except:
                sample['_original'] = str(sample['_original'])
        serializable.append(sample)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

def load_samples_from_cache(domain: str) -> list:
    cache_path = get_cache_path(domain)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cleanup_temp_folders():
    for pattern in ['temp_chroma_*', 'chroma_db_*']:
        for folder in glob.glob(pattern):
            try:
                shutil.rmtree(folder)
            except:
                pass

def get_ground_truth_answer(sample: dict) -> str:
    original = sample.get('_original', {})
    if isinstance(original, dict):
        for key in ['response', 'answer', 'ground_truth']:
            if key in original and original[key]:
                return str(original[key])
    for key in ['response', 'answer', 'ground_truth']:
        if key in sample and sample[key]:
            return str(sample[key])
    return ""

def extract_text_from_result(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, 'content'):
        return str(result.content)
    if hasattr(result, 'text'):
        return str(result.text)
    if isinstance(result, dict):
        return result.get('content', result.get('text', str(result)))
    return str(result)

def remove_thinking_tags(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reflection>.*?</reflection>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_api_key() -> str:
    for i in range(1, 21):
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key:
            return key
    return os.getenv('GROQ_API_KEY')

# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def init_components():
    """Initialize heavy components once."""
    loader = MultiDomainLoader()
    chunker = DocumentChunker()
    evaluator = TRACeEvaluator(judge_model=JUDGE_MODEL, use_local=True, verbose=False)
    api_key = get_api_key()
    groq_client = Groq(api_key=api_key) if api_key else None
    return loader, chunker, evaluator, groq_client

@st.cache_data
def load_domain_samples(_loader, domain: str) -> list:
    """Load samples for a domain with caching."""
    cached = load_samples_from_cache(domain)
    if cached:
        return cached

    try:
        data = _loader.load_domain(domain, "test")
        random.seed(RANDOM_SEED)
        indices = list(range(len(data)))
        random.shuffle(indices)
        samples = []
        for i in indices[:NUM_QUESTIONS_PER_DOMAIN]:
            sample = dict(data[i])
            sample['dataset_id'] = sample.get('id', f'{domain}_{i}')
            samples.append(sample)
        save_samples_to_cache(domain, samples)
        return samples
    except Exception as e:
        st.error(f"Error loading {domain}: {e}")
        return []

# ============================================================================
# RAG PIPELINE
# ============================================================================

def run_rag_pipeline(domain, sample, chunking, retrieval_type, embedding_method, llm_model,
                     chunker, evaluator, groq_client):
    """Run the RAG pipeline."""

    question = sample.get('question', '')
    ground_truth = get_ground_truth_answer(sample)
    documents = sample.get('documents', sample.get('context', []))

    if isinstance(documents, str):
        documents = [documents]
    corpus = "\n\n".join(documents) if documents else ""

    if not corpus:
        return {"error": "No documents found"}

    # 1. Chunking
    try:
        if chunking == "none":
            chunks = [corpus]
        elif chunking == "sentence":
            chunk_results = chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]
        else:  # semantic
            chunk_results = chunker.semantic_chunking(corpus)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]

        if not chunks:
            chunks = [corpus]
    except Exception as e:
        return {"error": f"Chunking error: {e}"}

    # 2. Retrieval
    cleanup_temp_folders()

    try:
        if retrieval_type == "sparse":
            retriever = UnifiedRetriever(method=embedding_method)
            retriever.index_documents(chunks=chunks)
        elif retrieval_type == "hybrid":
            retriever = HybridRetriever(
                dense_model=embedding_method,
                sparse_method="tfidf",
                alpha=0.5,
                verbose=False
            )
            retriever.index(chunks)
        else:  # dense
            temp_dir = f"./temp_chroma_{uuid.uuid4().hex[:8]}"
            retriever = UnifiedRetriever(
                method=embedding_method,
                persist_directory=temp_dir,
                collection_prefix="demo"
            )
            retriever.index_documents(chunks=chunks, clear_existing=True)

        results = retriever.retrieve(question, top_k=TOP_K)
        retrieved_chunks = [extract_text_from_result(r) for r in results] if results else chunks[:TOP_K]
    except Exception as e:
        cleanup_temp_folders()
        return {"error": f"Retrieval error: {e}", "ground_truth": ground_truth}

    cleanup_temp_folders()
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 3. Generation
    prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        generated_answer = response.choices[0].message.content.strip()
        generated_answer = remove_thinking_tags(generated_answer)
    except Exception as e:
        return {"error": f"LLM error: {e}", "ground_truth": ground_truth, "context": context}

    # 4. Evaluation
    try:
        result = evaluator.evaluate_single(
            question=question,
            ground_truth=ground_truth,
            generated_response=generated_answer,
            retrieved_chunks=retrieved_chunks
        )
        scores = result.trace_scores

        return {
            "answer": generated_answer,
            "relevance": scores.context_relevance,
            "utilization": scores.context_utilization,
            "completeness": scores.completeness,
            "adherence": scores.adherence_score(),
            "ground_truth": ground_truth,
            "context": context
        }
    except Exception as e:
        return {
            "answer": generated_answer,
            "relevance": 0, "utilization": 0, "completeness": 0, "adherence": 0,
            "ground_truth": ground_truth,
            "context": context,
            "eval_error": str(e)
        }

# ============================================================================
# MAIN UI
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header"><span>🔍 RAG Pipeline Demo</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Test different RAG configurations across multiple domains</p>', unsafe_allow_html=True)

    # Initialize components
    loader, chunker, evaluator, groq_client = init_components()

    if not groq_client:
        st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable.")
        return

    # Layout
    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.markdown('<div class="section-title">📋 Configuration</div>', unsafe_allow_html=True)

        # Domain selection
        domain = st.selectbox(
            "Domain",
            options=DOMAINS,
            format_func=lambda x: DOMAIN_NAMES[x],
            help="Select the domain/dataset"
        )

        # Load samples for selected domain
        samples = load_domain_samples(loader, domain)

        # Question selection
        question_options = {
            f"[{s.get('dataset_id', i)}] {s.get('question', '')[:70]}...": i
            for i, s in enumerate(samples)
        }

        selected_question = st.selectbox(
            "Question",
            options=list(question_options.keys()),
            help="Select a question from the dataset"
        )

        st.markdown('<div class="section-title" style="margin-top: 15px;">⚙️ RAG Components</div>', unsafe_allow_html=True)

        # RAG components
        chunking = st.selectbox("Chunking Strategy", CHUNKING_STRATEGIES, index=2, help="How to split documents")
        retrieval_type = st.selectbox("Retrieval Type", RETRIEVAL_TYPES, index=1, help="dense / sparse / hybrid")

        # Dynamic embedding options
        if retrieval_type == "sparse":
            embedding_options = SPARSE_METHODS
        else:
            embedding_options = DOMAIN_EMBEDDINGS.get(domain, ["minilm", "mpnet"])

        embedding_method = st.selectbox("Embedding / Method", embedding_options, help="Model or sparse method")
        llm_model = st.selectbox("LLM Model", LLM_MODELS, index=2, help="Model for generation")

        # Run button
        run_clicked = st.button("🚀 Run RAG Pipeline", use_container_width=True)

    with col_results:
        st.markdown('<div class="section-title">📊 Results</div>', unsafe_allow_html=True)

        # Dataset ID and Timer row (inline)
        id_timer_placeholder = st.empty()

        # Status message placeholder
        status_placeholder = st.empty()

        # Always show metrics (with default values)
        metrics_placeholder = st.empty()

        # Initialize with default values
        if 'results' not in st.session_state:
            st.session_state.results = {
                'dataset_id': '-',
                'relevance': 0.0,
                'utilization': 0.0,
                'completeness': 0.0,
                'adherence': 0.0,
                'answer': '',
                'ground_truth': '',
                'context': '',
                'elapsed_time': None
            }

        # Get current values
        res = st.session_state.results

        # Show dataset ID and timer row
        timer_html = ''
        if res.get('elapsed_time') is not None:
            timer_html = f'<span class="timer-box completed">✅ {res["elapsed_time"]:.1f}s</span>'

        id_timer_placeholder.markdown(f"""
        <div class="id-timer-row">
            <span class="dataset-id">Dataset ID: {res["dataset_id"]}</span>
            {timer_html}
        </div>
        """, unsafe_allow_html=True)

        # Show metrics
        metrics_placeholder.markdown(f"""
        <div style="display: flex; gap: 15px; margin-bottom: 15px;">
            <div class="metric-card" style="flex: 1;">
                <div class="metric-value metric-r">{res['relevance']:.3f}</div>
                <div class="metric-label">Relevance (R)</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-value metric-u">{res['utilization']:.3f}</div>
                <div class="metric-label">Utilization (U)</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-value metric-c">{res['completeness']:.3f}</div>
                <div class="metric-label">Completeness (C)</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-value metric-a">{res['adherence']:.3f}</div>
                <div class="metric-label">Adherence (A)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Answer boxes
        answer_placeholder = st.empty()
        answer_placeholder.markdown(f"""
        <div class="box-label">💡 Generated Answer</div>
        <div class="result-box answer-box">{res['answer'] if res['answer'] else '<span style="color: #64748b; font-style: italic;">Answer will appear here...</span>'}</div>
        """, unsafe_allow_html=True)

        ground_truth_placeholder = st.empty()
        ground_truth_placeholder.markdown(f"""
        <div class="box-label">✅ Ground Truth Answer</div>
        <div class="result-box ground-truth-box">{res['ground_truth'] if res['ground_truth'] else '<span style="color: #64748b; font-style: italic;">Ground truth will appear here...</span>'}</div>
        """, unsafe_allow_html=True)

        # Context expander
        context_placeholder = st.empty()
        with context_placeholder.expander("📁 Retrieved Context", expanded=False):
            if res['context']:
                st.markdown(f'<div class="result-box context-box">{res["context"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box context-box"><span style="color: #64748b; font-style: italic;">Context will appear here...</span></div>', unsafe_allow_html=True)

        # Handle run button click
        if run_clicked and selected_question:
            sample_idx = question_options[selected_question]
            sample = samples[sample_idx]
            dataset_id = sample.get('dataset_id', f'{domain}_{sample_idx}')

            question = sample.get('question', '')
            ground_truth = get_ground_truth_answer(sample)
            documents = sample.get('documents', sample.get('context', []))

            if isinstance(documents, str):
                documents = [documents]
            corpus = "\n\n".join(documents) if documents else ""

            # Start timer
            start_time = time.time()

            def update_timer_display(status_text, completed=False):
                elapsed = time.time() - start_time
                if completed:
                    timer_html = f'<span class="timer-box completed">✅ {elapsed:.1f}s</span>'
                else:
                    timer_html = f'<span class="timer-box">⏱️ {elapsed:.1f}s</span>'

                id_timer_placeholder.markdown(f"""
                <div class="id-timer-row">
                    <span class="dataset-id">Dataset ID: {dataset_id}</span>
                    {timer_html}
                </div>
                """, unsafe_allow_html=True)

                if status_text and not completed:
                    status_placeholder.info(status_text)
                else:
                    status_placeholder.empty()

            # Initial update
            update_timer_display("🔄 Starting pipeline...")

            error_result = None

            # Step 1: Chunking
            update_timer_display("📄 Chunking documents...")
            try:
                if chunking == "none":
                    chunks = [corpus]
                elif chunking == "sentence":
                    chunk_results = chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
                    chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]
                else:  # semantic
                    chunk_results = chunker.semantic_chunking(corpus)
                    chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]

                if not chunks:
                    chunks = [corpus]
            except Exception as e:
                error_result = f"Chunking error: {e}"

            # Step 2: Retrieval
            if not error_result:
                update_timer_display("🔍 Retrieving relevant chunks...")
                cleanup_temp_folders()

                try:
                    if retrieval_type == "sparse":
                        retriever = UnifiedRetriever(method=embedding_method)
                        retriever.index_documents(chunks=chunks)
                    elif retrieval_type == "hybrid":
                        retriever = HybridRetriever(
                            dense_model=embedding_method,
                            sparse_method="tfidf",
                            alpha=0.5,
                            verbose=False
                        )
                        retriever.index(chunks)
                    else:  # dense
                        temp_dir = f"./temp_chroma_{uuid.uuid4().hex[:8]}"
                        retriever = UnifiedRetriever(
                            method=embedding_method,
                            persist_directory=temp_dir,
                            collection_prefix="demo"
                        )
                        retriever.index_documents(chunks=chunks, clear_existing=True)

                    results = retriever.retrieve(question, top_k=TOP_K)
                    retrieved_chunks = [extract_text_from_result(r) for r in results] if results else chunks[:TOP_K]
                except Exception as e:
                    cleanup_temp_folders()
                    error_result = f"Retrieval error: {e}"

                cleanup_temp_folders()

            # Step 3: Generation
            if not error_result:
                update_timer_display("🤖 Generating answer with LLM...")
                context = "\n\n---\n\n".join(retrieved_chunks)

                prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

                try:
                    response = groq_client.chat.completions.create(
                        model=llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.1
                    )
                    generated_answer = response.choices[0].message.content.strip()
                    generated_answer = remove_thinking_tags(generated_answer)
                except Exception as e:
                    error_result = f"LLM error: {e}"

            # Step 4: Evaluation
            if not error_result:
                update_timer_display("📊 Evaluating with TRACe metrics...")

                try:
                    result = evaluator.evaluate_single(
                        question=question,
                        ground_truth=ground_truth,
                        generated_response=generated_answer,
                        retrieved_chunks=retrieved_chunks
                    )
                    scores = result.trace_scores
                    r, u, c, a = scores.context_relevance, scores.context_utilization, scores.completeness, scores.adherence_score()
                except Exception as e:
                    r = u = c = a = 0.0

            # Final update
            elapsed = time.time() - start_time
            update_timer_display("", completed=True)

            if error_result:
                st.error(f"❌ {error_result}")
            else:
                # Update session state with results
                st.session_state.results = {
                    'dataset_id': dataset_id,
                    'relevance': r,
                    'utilization': u,
                    'completeness': c,
                    'adherence': a,
                    'answer': generated_answer,
                    'ground_truth': ground_truth,
                    'context': context,
                    'elapsed_time': elapsed
                }

                # Update all displays
                metrics_placeholder.markdown(f"""
                <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value metric-r">{r:.3f}</div>
                        <div class="metric-label">Relevance (R)</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value metric-u">{u:.3f}</div>
                        <div class="metric-label">Utilization (U)</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value metric-c">{c:.3f}</div>
                        <div class="metric-label">Completeness (C)</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value metric-a">{a:.3f}</div>
                        <div class="metric-label">Adherence (A)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                answer_placeholder.markdown(f"""
                <div class="box-label">💡 Generated Answer</div>
                <div class="result-box answer-box">{generated_answer}</div>
                """, unsafe_allow_html=True)

                ground_truth_placeholder.markdown(f"""
                <div class="box-label">✅ Ground Truth Answer</div>
                <div class="result-box ground-truth-box">{ground_truth}</div>
                """, unsafe_allow_html=True)

                with context_placeholder.expander("📁 Retrieved Context", expanded=False):
                    st.markdown(f'<div class="result-box context-box">{context}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <b>Metrics:</b>
        <span style="color: #34d399;">● R</span> Relevance |
        <span style="color: #60a5fa;">● U</span> Utilization |
        <span style="color: #fbbf24;">● C</span> Completeness |
        <span style="color: #f472b6;">● A</span> Adherence
        (Higher is better, 0-1)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()