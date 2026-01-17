"""
RAG Pipeline Demo UI
Interactive interface to test RAG configurations across domains.

Usage:
    python rag_demo_ui.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import random
import uuid
import shutil
import glob
import json
from pathlib import Path
import gradio as gr
import numpy as np

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.multidomain_loader import MultiDomainLoader
from src.chunker import DocumentChunker
from src.evaluator import TRACeEvaluator
from src.retriever import UnifiedRetriever
from src.hybrid_retriever import HybridRetriever
from groq import Groq

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TOP_K = 5
NUM_QUESTIONS_PER_DOMAIN = 10
CACHE_DIR = Path("./dataset_cache")

# Judge model (local Ollama)
JUDGE_MODEL = "qwen2.5:7b-instruct"

# Domains
DOMAINS = ["finqa", "cuad", "delucionqa", "hotpotqa", "covidqa"]

DOMAIN_NAMES = {
    "finqa": "Finance (FinQA)",
    "cuad": "Legal (CUAD)",
    "delucionqa": "Customer Support (DelucionQA)",
    "hotpotqa": "General Knowledge (HotpotQA)",
    "covidqa": "Biomedical (CovidQA)"
}

# RAG Component Options
CHUNKING_STRATEGIES = ["none", "sentence", "semantic"]
RETRIEVAL_TYPES = ["dense", "sparse", "hybrid"]

# Embedding models per domain
DOMAIN_EMBEDDINGS = {
    "finqa": ["finbert", "bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "cuad": ["legal-bert", "e5-large", "bge-large", "minilm", "mpnet"],
    "delucionqa": ["bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "hotpotqa": ["bge-large", "e5-large", "gte-large", "minilm", "mpnet"],
    "covidqa": ["biobert", "sapbert", "pubmedbert", "bge-base", "minilm", "mpnet"]
}

# Sparse methods
SPARSE_METHODS = ["bm25", "tfidf"]

# LLM models (from Groq)
LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "groq/compound"
]


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

def get_cache_path(domain: str) -> Path:
    return CACHE_DIR / f"{domain}_samples.json"


def save_samples_to_cache(domain: str, samples: list):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = get_cache_path(domain)

    serializable_samples = []
    for sample in samples:
        s = dict(sample)
        if '_original' in s:
            try:
                s['_original'] = dict(s['_original']) if hasattr(s['_original'], 'items') else str(s['_original'])
            except:
                s['_original'] = str(s['_original'])
        serializable_samples.append(s)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_samples, f, ensure_ascii=False, indent=2)
    print(f"    💾 Cached {len(samples)} samples to {cache_path}")


def load_samples_from_cache(domain: str) -> list:
    cache_path = get_cache_path(domain)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"    ✅ Loaded {len(samples)} samples from cache")
        return samples
    return None


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.loader = MultiDomainLoader()
        self.chunker = DocumentChunker()
        self.trace_evaluator = None
        self.groq_client = None
        self.domain_samples = {}
        self.current_sample = None

    def initialize(self):
        print("🔧 Initializing RAG Demo UI...")

        # Initialize Groq client
        api_key = self._get_api_key()
        if api_key:
            self.groq_client = Groq(api_key=api_key)
            print("  ✅ Groq API initialized")
        else:
            print("  ⚠️ No Groq API key found")

        # Initialize judge model
        print(f"  🔧 Initializing judge model: {JUDGE_MODEL}")
        self.trace_evaluator = TRACeEvaluator(
            judge_model=JUDGE_MODEL,
            use_local=True,
            verbose=False
        )
        print("  ✅ Judge model ready")

        # Load samples from each domain
        print("  📂 Loading samples from domains...")
        random.seed(RANDOM_SEED)

        for domain in DOMAINS:
            # Try cache first
            cached = load_samples_from_cache(domain)
            if cached:
                self.domain_samples[domain] = cached
                continue

            # Download and cache
            try:
                print(f"    ⬇️ Downloading {domain}...")
                data = self.loader.load_domain(domain, "test")
                indices = list(range(len(data)))
                random.shuffle(indices)
                samples = []
                for i in indices[:NUM_QUESTIONS_PER_DOMAIN]:
                    sample = dict(data[i])
                    sample['dataset_id'] = sample.get('id', f'{domain}_{i}')
                    samples.append(sample)
                self.domain_samples[domain] = samples
                save_samples_to_cache(domain, samples)
            except Exception as e:
                print(f"    ❌ {domain}: {e}")
                self.domain_samples[domain] = []

        print("✅ Initialization complete!")

    def _get_api_key(self) -> str:
        for i in range(1, 21):
            key = os.getenv(f'GROQ_API_KEY_{i}')
            if key:
                return key
        return os.getenv('GROQ_API_KEY')


state = AppState()


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


def get_questions_for_domain(domain: str) -> list:
    samples = state.domain_samples.get(domain, [])
    questions = []
    for i, s in enumerate(samples):
        q = s.get('question', '')
        dataset_id = s.get('dataset_id', s.get('id', f'{domain}_{i}'))
        display = q[:80] + "..." if len(q) > 80 else q
        questions.append(f"[{dataset_id}] {display}")
    return questions


def get_embedding_options(domain: str, retrieval_type: str) -> list:
    if retrieval_type == "sparse":
        return SPARSE_METHODS
    else:
        return DOMAIN_EMBEDDINGS.get(domain, ["minilm", "mpnet"])


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
    """Safely extract text from retrieval result."""
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
    """Remove thinking/reasoning tags from model output."""
    import re
    # Remove <think>...</think> tags and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove <thinking>...</thinking> tags and content
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove <reasoning>...</reasoning> tags and content
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove <reflection>...</reflection> tags and content
    text = re.sub(r'<reflection>.*?</reflection>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================================
# RAG PIPELINE
# ============================================================================

def run_rag_pipeline(
        domain: str,
        question_idx: str,
        chunking: str,
        retrieval_type: str,
        embedding_method: str,
        llm_model: str,
        progress=gr.Progress()
) -> tuple:
    """Run the RAG pipeline with selected configuration."""

    if not state.groq_client:
        return "❌ Groq API not initialized.", "", "0.000", "0.000", "0.000", "0.000", "", ""

    if not question_idx:
        return "❌ Please select a question.", "", "0.000", "0.000", "0.000", "0.000", "", ""

    # Extract dataset_id from "[dataset_id] question text"
    import re
    match = re.match(r'\[([^\]]+)\]', question_idx)
    if match:
        dataset_id = match.group(1)
    else:
        dataset_id = "N/A"

    # Find sample by dataset_id
    samples = state.domain_samples.get(domain, [])
    sample = None
    for s in samples:
        sid = s.get('dataset_id', s.get('id', ''))
        if str(sid) == str(dataset_id):
            sample = s
            break

    if not sample:
        # Fallback to index-based lookup
        try:
            idx = int(question_idx.split("]")[0].replace("[", "")) - 1
            if 0 <= idx < len(samples):
                sample = samples[idx]
        except:
            pass

    if not sample:
        return "❌ Sample not found.", dataset_id, "0.000", "0.000", "0.000", "0.000", "", ""

    question = sample.get('question', '')
    ground_truth = get_ground_truth_answer(sample)
    documents = sample.get('documents', sample.get('context', []))

    if isinstance(documents, str):
        documents = [documents]
    corpus = "\n\n".join(documents) if documents else ""

    if not corpus:
        return "❌ No documents found.", dataset_id, "0.000", "0.000", "0.000", "0.000", "", ""

    progress(0.1, desc="Chunking documents...")

    # 1. Chunking
    try:
        if chunking == "none":
            chunks = [corpus]
        elif chunking == "sentence":
            chunk_results = state.chunker.sentence_chunking(corpus, chunk_size=5, overlap=2)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]
        else:  # semantic
            chunk_results = state.chunker.semantic_chunking(corpus)
            chunks = [c['text'] for c in chunk_results] if chunk_results else [corpus]

        if not chunks:
            chunks = [corpus]
    except Exception as e:
        return f"❌ Chunking error: {e}", dataset_id, "0.000", "0.000", "0.000", "0.000", "", ""

    progress(0.3, desc="Retrieving relevant chunks...")

    # 2. Retrieval
    cleanup_temp_folders()
    retrieved_chunks = []

    try:
        if retrieval_type == "sparse":
            retriever = UnifiedRetriever(method=embedding_method)
            retriever.index_documents(chunks=chunks)
            results = retriever.retrieve(question, top_k=TOP_K)
        elif retrieval_type == "hybrid":
            retriever = HybridRetriever(
                dense_model=embedding_method,
                sparse_method="tfidf",
                alpha=0.5,
                verbose=False
            )
            retriever.index(chunks)
            results = retriever.retrieve(question, top_k=TOP_K)
        else:  # dense
            temp_dir = f"./temp_chroma_{uuid.uuid4().hex[:8]}"
            retriever = UnifiedRetriever(
                method=embedding_method,
                persist_directory=temp_dir,
                collection_prefix="demo"
            )
            retriever.index_documents(chunks=chunks, clear_existing=True)
            results = retriever.retrieve(question, top_k=TOP_K)

        # Extract text from results
        if results:
            retrieved_chunks = [extract_text_from_result(r) for r in results]
        else:
            retrieved_chunks = chunks[:TOP_K]

    except Exception as e:
        cleanup_temp_folders()
        return f"❌ Retrieval error: {e}", dataset_id, "0.000", "0.000", "0.000", "0.000", ground_truth, ""

    cleanup_temp_folders()
    context = "\n\n---\n\n".join(retrieved_chunks)

    progress(0.5, desc="Generating answer with LLM...")

    # 3. Generation
    prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = state.groq_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        generated_answer = response.choices[0].message.content.strip()
        # Remove thinking tags from reasoning models
        generated_answer = remove_thinking_tags(generated_answer)
    except Exception as e:
        return f"❌ LLM error: {e}", dataset_id, "0.000", "0.000", "0.000", "0.000", ground_truth, context

    progress(0.7, desc="Evaluating with TRACe metrics...")

    # 4. Evaluation
    try:
        result = state.trace_evaluator.evaluate_single(
            question=question,
            ground_truth=ground_truth,
            generated_response=generated_answer,
            retrieved_chunks=retrieved_chunks
        )
        scores = result.trace_scores

        relevance = f"{scores.context_relevance:.3f}"
        utilization = f"{scores.context_utilization:.3f}"
        completeness = f"{scores.completeness:.3f}"
        adherence = f"{scores.adherence_score():.3f}"

    except Exception as e:
        relevance = utilization = completeness = adherence = "Error"

    progress(1.0, desc="Done!")
    return generated_answer, dataset_id, relevance, utilization, completeness, adherence, ground_truth, context


# ============================================================================
# UI CALLBACKS
# ============================================================================

def on_domain_change(domain):
    questions = get_questions_for_domain(domain)
    embeddings = get_embedding_options(domain, "dense")
    return (
        gr.update(choices=questions, value=questions[0] if questions else None),
        gr.update(choices=embeddings, value=embeddings[0] if embeddings else None)
    )


def on_retrieval_type_change(retrieval_type, domain):
    options = get_embedding_options(domain, retrieval_type)
    return gr.update(choices=options, value=options[0] if options else None)


# ============================================================================
# BUILD UI - DARK THEME
# ============================================================================

def create_ui():
    with gr.Blocks(
        title="RAG Pipeline Demo",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0f172a",
            block_background_fill="#1e293b",
            block_border_color="#334155",
            block_label_text_color="#e2e8f0",
            body_text_color="#e2e8f0",
            input_background_fill="#0f172a",
            input_border_color="#475569",
            button_primary_background_fill="#6366f1",
            button_primary_background_fill_hover="#818cf8",
            button_primary_text_color="#ffffff",
        ),
        css="""
            .gradio-container { max-width: 1400px !important; margin: auto !important; padding: 0 !important; }
            .gr-block, .gr-box { padding: 8px !important; margin: 4px 0 !important; }
            .gr-form { gap: 6px !important; }
            .gr-padded { padding: 8px !important; }
            
            .header-title {
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2rem !important;
                font-weight: 800 !important;
                margin: 0 !important;
            }
            .header-subtitle { color: #94a3b8 !important; font-size: 0.95rem !important; margin: 4px 0 0 0 !important; }
            
            .section-title {
                color: #f8fafc !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                margin-bottom: 8px !important;
            }
            
            /* Labels and info text */
            label, .label-wrap > span { color: #c7d2fe !important; font-weight: 600 !important; font-size: 0.8rem !important; }
            span.svelte-1gfkn6j { color: #a5b4fc !important; font-size: 0.75rem !important; }
            
            /* Dropdowns */
            .gr-dropdown, select, .wrap.svelte-1cl284s {
                background: #0f172a !important;
                border: 2px solid #475569 !important;
                border-radius: 8px !important;
                color: #e2e8f0 !important;
                padding: 6px 10px !important;
                min-height: 36px !important;
            }
            .gr-dropdown:hover { border-color: #6366f1 !important; }
            
            /* Dropdown options list */
            ul[role="listbox"], .options, div[role="listbox"] {
                background: #1e293b !important;
                border: 2px solid #475569 !important;
                border-radius: 10px !important;
            }
            ul[role="listbox"] li, .options .option, div[role="option"], li[role="option"] {
                background: #1e293b !important;
                color: #e2e8f0 !important;
                padding: 10px 14px !important;
                border-bottom: 1px solid #334155 !important;
            }
            ul[role="listbox"] li:hover, .options .option:hover, div[role="option"]:hover, li[role="option"]:hover {
                background: #334155 !important;
                color: #ffffff !important;
            }
            ul[role="listbox"] li[aria-selected="true"], li[role="option"][aria-selected="true"] {
                background: #4f46e5 !important;
                color: #ffffff !important;
            }
            
            /* Question dropdown - single line */
            #question-dropdown .wrap { min-height: 36px !important; }
            #question-dropdown input, #question-dropdown .single-select {
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                line-height: 1.3 !important;
                min-height: 36px !important;
                padding: 8px !important;
            }
            #question-dropdown .options { max-height: 350px !important; background: #1e293b !important; }
            #question-dropdown .option {
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                padding: 10px !important;
                line-height: 1.3 !important;
                border-bottom: 1px solid #334155 !important;
            }
            #question-dropdown .option:hover { background: #334155 !important; }
            
            /* Dataset ID */
            #dataset-id-box { min-height: 40px !important; }
            #dataset-id-box input {
                background: linear-gradient(135deg, #312e81 0%, #4c1d95 100%) !important;
                border: 2px solid #6366f1 !important;
                color: #e0e7ff !important;
                font-family: monospace !important;
                font-size: 0.9rem !important;
                font-weight: 600 !important;
                text-align: center !important;
                border-radius: 6px !important;
                padding: 4px !important;
            }
            
            /* Metric cards */
            .metric-card { min-height: 50px !important; }
            .metric-card input {
                background: transparent !important;
                border: none !important;
                font-size: 1.3rem !important;
                font-weight: 700 !important;
                text-align: center !important;
                font-family: monospace !important;
                padding: 4px !important;
            }
            #metric-r input { color: #34d399 !important; }
            #metric-u input { color: #60a5fa !important; }
            #metric-c input { color: #fbbf24 !important; }
            #metric-a input { color: #f472b6 !important; }
            
            /* Text areas */
            textarea {
                background: #0f172a !important;
                border: 2px solid #334155 !important;
                border-radius: 8px !important;
                color: #e2e8f0 !important;
                line-height: 1.5 !important;
                padding: 10px !important;
                font-size: 0.85rem !important;
            }
            textarea:hover { border-color: #475569 !important; }
            textarea:focus { border-color: #6366f1 !important; }
            
            #answer-box textarea { border-color: #4f46e5 !important; min-height: 80px !important; }
            #ground-truth-box textarea { border-color: #22c55e !important; min-height: 80px !important; }
            #context-box textarea { min-height: 60px !important; font-size: 0.8rem !important; }
            
            /* Submit button */
            .submit-btn {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                padding: 10px 20px !important;
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
            }
            .submit-btn:hover { box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important; }
            
            .footer-text { color: #64748b !important; font-size: 0.8rem !important; text-align: center; margin-top: 8px !important; padding: 8px !important; }
            .footer-text b { color: #94a3b8 !important; }
            
            /* Accordion compact */
            .gr-accordion { margin: 4px 0 !important; }
            .gr-accordion .label-wrap { padding: 8px !important; }
        """
    ) as demo:

        gr.HTML("""
            <div style="text-align: center; padding: 12px 0 8px 0;">
                <h1 class="header-title">🔍 RAG Pipeline Demo</h1>
                <p class="header-subtitle">Test different RAG configurations across multiple domains</p>
            </div>
        """)

        with gr.Row():
            # Left - Configuration
            with gr.Column(scale=1):
                gr.HTML('<div class="section-title">📋 Configuration</div>')

                domain_dropdown = gr.Dropdown(
                    choices=[(DOMAIN_NAMES[d], d) for d in DOMAINS],
                    value="finqa",
                    label="Domain",
                    info="Select the domain/dataset"
                )

                question_dropdown = gr.Dropdown(
                    choices=get_questions_for_domain("finqa"),
                    value=get_questions_for_domain("finqa")[0] if get_questions_for_domain("finqa") else None,
                    label="Question",
                    info="Select a question from the dataset",
                    elem_id="question-dropdown"
                )

                gr.HTML('<div class="section-title" style="margin-top: 10px;">⚙️ RAG Components</div>')

                chunking_dropdown = gr.Dropdown(
                    choices=CHUNKING_STRATEGIES,
                    value="semantic",
                    label="Chunking Strategy",
                    info="How to split documents"
                )

                retrieval_type_dropdown = gr.Dropdown(
                    choices=RETRIEVAL_TYPES,
                    value="sparse",
                    label="Retrieval Type",
                    info="dense / sparse / hybrid"
                )

                embedding_dropdown = gr.Dropdown(
                    choices=SPARSE_METHODS,
                    value="bm25",
                    label="Embedding / Method",
                    info="Model or sparse method"
                )

                llm_dropdown = gr.Dropdown(
                    choices=LLM_MODELS,
                    value="qwen/qwen3-32b",
                    label="LLM Model",
                    info="Model for generation"
                )

                submit_btn = gr.Button("🚀 Run RAG Pipeline", variant="primary", size="lg", elem_classes=["submit-btn"])

            # Right - Results
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">📊 Results</div>')

                dataset_id_box = gr.Textbox(label="Dataset ID", interactive=False, elem_id="dataset-id-box")

                with gr.Row():
                    relevance_box = gr.Textbox(label="Relevance (R)", value="0.000", interactive=False, elem_id="metric-r", elem_classes=["metric-card"])
                    utilization_box = gr.Textbox(label="Utilization (U)", value="0.000", interactive=False, elem_id="metric-u", elem_classes=["metric-card"])
                    completeness_box = gr.Textbox(label="Completeness (C)", value="0.000", interactive=False, elem_id="metric-c", elem_classes=["metric-card"])
                    adherence_box = gr.Textbox(label="Adherence (A)", value="0.000", interactive=False, elem_id="metric-a", elem_classes=["metric-card"])

                answer_box = gr.Textbox(label="💡 Generated Answer", interactive=False, lines=3, elem_id="answer-box")
                ground_truth_box = gr.Textbox(label="✅ Ground Truth Answer", interactive=False, lines=3, elem_id="ground-truth-box")

                with gr.Accordion("📁 Retrieved Context", open=False):
                    context_box = gr.Textbox(label="", interactive=False, lines=4, elem_id="context-box")

        gr.HTML("""
            <div class="footer-text">
                <b>Metrics:</b>
                <span style="color: #34d399;">● R</span> Relevance |
                <span style="color: #60a5fa;">● U</span> Utilization |
                <span style="color: #fbbf24;">● C</span> Completeness |
                <span style="color: #f472b6;">● A</span> Adherence
                (Higher is better, 0-1)
            </div>
        """)

        # Event handlers
        domain_dropdown.change(fn=on_domain_change, inputs=[domain_dropdown], outputs=[question_dropdown, embedding_dropdown])
        retrieval_type_dropdown.change(fn=on_retrieval_type_change, inputs=[retrieval_type_dropdown, domain_dropdown], outputs=[embedding_dropdown])

        submit_btn.click(
            fn=run_rag_pipeline,
            inputs=[domain_dropdown, question_dropdown, chunking_dropdown, retrieval_type_dropdown, embedding_dropdown, llm_dropdown],
            outputs=[answer_box, dataset_id_box, relevance_box, utilization_box, completeness_box, adherence_box, ground_truth_box, context_box]
        )

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RAG Pipeline Demo UI")
    print("=" * 60)

    state.initialize()
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)