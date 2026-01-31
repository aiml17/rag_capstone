# RAG Capstone Project

A comprehensive **Retrieval-Augmented Generation (RAG)** pipeline for multi-domain question answering using the **RAGBench** dataset. This project implements and benchmarks various RAG configurations across embedding models, chunking strategies, retrieval methods, and LLM generators.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Supported Domains](#-supported-domains)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [1. Verify System Setup](#1-verify-system-setup)
  - [2. Run Preliminary Evaluation](#2-run-preliminary-evaluation)
  - [3. Run Final Evaluation](#3-run-final-evaluation)
  - [4. Launch Demo UI](#4-launch-demo-ui)
- [Project Structure](#-project-structure)
- [Evaluation Metrics](#-evaluation-metrics)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **Multi-Domain Support**: Evaluate across 5 RAGBench domains (Finance, Legal, Customer Support, General Knowledge, Biomedical)
- **Multiple Embedding Models**: Compare 15+ embedding models including domain-specific ones (FinBERT, LegalBERT, BioBERT)
- **Flexible Retrieval**: Dense, Sparse (BM25/TF-IDF), and Hybrid retrieval strategies
- **Chunking Strategies**: None, Sentence-based, and Semantic chunking
- **LLM Integration**: Multiple LLM models via Groq API (Llama 3.1/3.3/4, Qwen 3, DeepSeek)
- **TRACe Evaluation**: Industry-standard RAG evaluation metrics from RAGBench paper
- **Apple Silicon Optimized**: MPS (Metal Performance Shaders) support for M-series chips
- **Checkpointing**: Resume interrupted evaluations seamlessly
- **Interactive Demo**: Streamlit and Gradio UI for testing configurations

---

## 🌐 Supported Domains

| Domain | Dataset | Description |
|--------|---------|-------------|
| **Finance** | FinQA | Financial Q&A from earnings reports and SEC filings |
| **Legal** | CUAD | Contract Understanding Atticus Dataset - legal clause extraction |
| **Customer Support** | DelucionQA | Technical customer support Q&A from IT domain |
| **General Knowledge** | HotpotQA | Multi-hop reasoning questions from Wikipedia |
| **Biomedical** | CovidQA | COVID-19 biomedical research questions |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline Flow                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Data    │───▶│  Chunker │───▶│ Retriever│───▶│Generator │          │
│  │  Loader  │    │          │    │          │    │  (LLM)   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       │              │               │                │                 │
│       ▼              ▼               ▼                ▼                 │
│  ┌──────────┐  ┌───────────┐  ┌───────────────┐  ┌──────────┐          │
│  │ RAGBench │  │ • None    │  │ • Dense       │  │ Groq API │          │
│  │ Dataset  │  │ • Sentence│  │ • Sparse      │  │ • Llama  │          │
│  │ (5 doms) │  │ • Semantic│  │ • Hybrid      │  │ • Qwen   │          │
│  └──────────┘  └───────────┘  └───────────────┘  └──────────┘          │
│                                      │                                  │
│                                      ▼                                  │
│                              ┌───────────────┐                          │
│                              │   ChromaDB    │                          │
│                              │ Vector Store  │                          │
│                              └───────────────┘                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

For detailed architecture diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📦 Prerequisites

- **Python**: 3.10 or higher
- **macOS**: Apple Silicon (M1/M2/M3/M4) recommended for MPS acceleration
- **API Keys**:
  - **Groq API Key** (required) - Get from [console.groq.com](https://console.groq.com)
  - **Ollama** (optional) - For local LLM-as-Judge evaluation
- **Disk Space**: ~5GB for models and vector stores

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd rag_capstone_new
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add your API keys to `.env`:

```env
# Groq API Key (Required)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Additional Groq API Keys for rotation (helps with rate limits)
GROQ_API_KEY_2=your_second_key_here
GROQ_API_KEY_3=your_third_key_here

# Optional: OpenAI API Key (for additional models)
OPENAI_API_KEY=your_openai_key_here

# Optional: HuggingFace Token (for gated models)
HF_TOKEN=your_huggingface_token_here
```

### Step 5: Install Ollama (Optional - for Local Judge)

For local LLM-as-Judge evaluation:

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull the judge model (in a new terminal)
ollama pull qwen2.5:7b-instruct

# For better quality (requires more VRAM)
ollama pull qwen2.5:32b-instruct
```

---

## ⚙️ Configuration

### Embedding Models Available

| Category | Models |
|----------|--------|
| **General Purpose** | `minilm`, `mpnet`, `bge-base`, `bge-large`, `e5-base`, `e5-large`, `gte-large` |
| **Finance** | `finbert` |
| **Legal** | `legal-bert` |
| **Biomedical** | `biobert`, `pubmedbert`, `sapbert` |

### LLM Models (via Groq)

| Model | Speed | Quality |
|-------|-------|---------|
| `llama-3.1-8b-instant` | ⚡ Fast | Good |
| `llama-3.3-70b-versatile` | Medium | Excellent |
| `llama-4-scout-17b-16e-instruct` | Medium | Very Good |
| `qwen-qwq-32b` | Medium | Excellent |
| `deepseek-r1-distill-llama-70b` | Slow | Best |

### Chunking Strategies

- **`none`**: No chunking, use full documents
- **`sentence`**: Fixed-size sentence chunks with configurable overlap
- **`semantic`**: Dynamic chunks based on semantic similarity

### Retrieval Types

- **`dense`**: Embedding-based semantic search
- **`sparse`**: Lexical search (BM25 or TF-IDF)
- **`hybrid`**: Combination of dense and sparse retrieval

---

## 📖 Usage

### 1. Verify System Setup

Test if your system is properly configured (especially Apple Silicon MPS):

```bash
python test_mps.py
```

**Expected Output:**
```
============================================================
SYSTEM INFORMATION
============================================================
Platform: macOS-...
Processor: arm
PyTorch version: 2.1.2

============================================================
MPS (METAL PERFORMANCE SHADERS) STATUS
============================================================
MPS available: True
MPS built: True
Successfully created tensor on MPS device!
✅ Your M4 Mac is ready for GPU-accelerated PyTorch!
```

---

### 2. Run Preliminary Evaluation

The preliminary evaluation tests all configurations with a small sample (10 examples each) to find the top performers.

#### List Available Domains and Configuration Counts

```bash
python src/run_preliminary_evaluation.py --list
```

#### Run Evaluation for a Specific Domain

```bash
# Run evaluation for Finance domain
python src/run_preliminary_evaluation.py --domain finqa --run

# Run evaluation for Legal domain
python src/run_preliminary_evaluation.py --domain cuad --run

# Run evaluation for Customer Support domain
python src/run_preliminary_evaluation.py --domain delucionqa --run

# Run evaluation for General Knowledge domain
python src/run_preliminary_evaluation.py --domain hotpotqa --run

# Run evaluation for Biomedical domain
python src/run_preliminary_evaluation.py --domain covidqa --run
```

#### Check Progress

```bash
python src/run_preliminary_evaluation.py --domain finqa --progress
```

#### View Results (Sorted by TRACe Score)

```bash
python src/run_preliminary_evaluation.py --domain finqa --results
```

#### Get Top 10 Configurations

```bash
python src/run_preliminary_evaluation.py --domain finqa --top10
```

---

### 3. Run Final Evaluation

The final evaluation tests the top 10 configurations with 100 examples each for statistical significance.

#### Run Final Evaluation

```bash
python src/run_final_evaluation.py --run
```

#### Check Progress

```bash
python src/run_final_evaluation.py --progress
```

#### View Results

```bash
python src/run_final_evaluation.py --results
```

---

### 4. Launch Demo UI

#### Streamlit UI (Recommended)

```bash
streamlit run src/rag_demo_streamlit.py
```

Then open your browser at: **http://localhost:8501**

#### Gradio UI (Alternative)

```bash
python src/rag_demo_ui.py
```

Then open your browser at: **http://localhost:7860**

---

## 📁 Project Structure

```
rag_capstone_new/
├── 📄 README.md                    # This file
├── 📄 ARCHITECTURE.md              # Detailed architecture documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env                         # Environment variables (API keys)
├── 📄 main.py                      # Entry point (placeholder)
├── 📄 test_mps.py                  # Apple Silicon MPS verification
│
├── 📂 src/                         # Source code
│   ├── 📄 data_loader.py           # FinQA dataset loader
│   ├── 📄 multidomain_loader.py    # Multi-domain RAGBench loader
│   ├── 📄 chunker.py               # Document chunking strategies
│   ├── 📄 retriever.py             # Dense/Sparse retrieval
│   ├── 📄 hybrid_retriever.py      # Hybrid retrieval implementation
│   ├── 📄 reranker.py              # Document reranking
│   ├── 📄 generator.py             # LLM response generation
│   ├── 📄 evaluator.py             # TRACe metrics evaluator
│   ├── 📄 pipeline.py              # Full RAG pipeline orchestration
│   ├── 📄 run_preliminary_evaluation.py  # Preliminary evaluation script
│   ├── 📄 run_final_evaluation.py        # Final evaluation script
│   ├── 📄 rag_demo_streamlit.py    # Streamlit demo UI
│   └── 📄 rag_demo_ui.py           # Gradio demo UI
│
├── 📂 data/                        # Data files
│   └── 📂 rgb/                     # RAGBench data cache
│
├── 📂 chroma_db/                   # ChromaDB vector store
├── 📂 checkpoints/                 # Evaluation checkpoints
├── 📂 preliminary_evaluations/     # Preliminary results
├── 📂 evaluations/                 # Final evaluation results
├── 📂 results/                     # Analysis results
└── 📂 dataset_cache/               # HuggingFace dataset cache
```

---

## 📊 Evaluation Metrics

This project uses **TRACe** metrics from the RAGBench paper:

| Metric | Description | Range |
|--------|-------------|-------|
| **Relevance** | How relevant are retrieved documents to the question? | 0-1 |
| **Utilization** | How much of the retrieved content is used in the answer? | 0-1 |
| **Completeness** | Does the answer fully address the question? | 0-1 |
| **Adherence** | Does the answer stick to the retrieved content (no hallucination)? | 0/1 |

### Aggregate Metrics

- **TRACe Score**: Average of Relevance, Utilization, Completeness (higher is better)
- **RMSE**: Root Mean Square Error for continuous metrics
- **AUROC**: Area Under ROC Curve for Adherence classification
- **F1 Score**: Harmonic mean of Precision and Recall for Adherence

---

## 🔧 Troubleshooting

### Rate Limit Errors (Groq)

If you encounter rate limit errors:

1. **Add more API keys** in `.env` file (supports key rotation)
2. **Increase delay** between calls in the evaluation script
3. **Wait and resume** - checkpointing saves progress automatically

### ChromaDB Telemetry Warnings

These are suppressed by default. If you see any, ensure environment variables are set:

```python
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
```

### Memory Issues

For large embedding models on limited RAM:

```bash
# Use smaller models
--embedding minilm  # Instead of bge-large

# Or reduce batch size in retriever.py
```

### MPS Not Available (macOS)

1. Ensure PyTorch is installed with MPS support:
   ```bash
   pip install torch==2.1.2
   ```
2. Update to macOS 12.3 or later
3. Fall back to CPU if needed (automatic)

### Ollama Connection Refused

```bash
# Ensure Ollama is running
ollama serve

# Check if model is pulled
ollama list
```

---

## 📚 References

- [RAGBench Paper](https://arxiv.org/abs/2407.11005) - TRACe evaluation methodology
- [Groq API Documentation](https://console.groq.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📝 License

This project is for educational and research purposes.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Happy RAG-ing! 🚀**
