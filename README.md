# AI Support Agent for @AppleSupport

An AI-powered customer support agent built on real Twitter support conversations (~3M tweets). The agent classifies customer intents, drafts contextual replies grounded in historical data, and decides whether to auto-handle or escalate to a human.

**Brand**: AppleSupport — chosen for high volume and diverse technical support issues (device troubleshooting, account/billing, software updates, hardware, connectivity).

## Architecture

```
Customer Tweet
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│   Intent     │     │  Historical Reply │
│  Classifier  │     │   Retriever (RAG) │
│  (LLM-based) │     │  (FAISS + embed)  │
└──────┬──────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────┐
│        Reply Generator           │
│  (Gemini with retrieved context) │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│      Escalation Decision         │
│  (rules + LLM hybrid)           │
└──────────────┬───────────────────┘
               │
               ▼
      Auto-reply OR Escalate
```

## Quick Start — Reproduce Headline Results

> **Fast path (~5 min)**: Steps 1-5 + `python eval/run_eval.py --baselines-only` runs intent and escalation baselines with zero API calls.
>
> **Full path (~15-20 min)**: All steps including LLM evaluation and reply generation. Gemini free tier rate limits may add wait time.

### Prerequisites
- Python 3.10+
- Google Gemini API key ([get one free](https://aistudio.google.com/apikey))
- ~500 MB disk space for dataset

### Step 1: Install Dependencies (~2 min)
```bash
git clone <repo-url> && cd hiver
pip install -r requirements.txt
```

**Dependencies**: `pandas`, `numpy`, `google-genai`, `scikit-learn`, `faiss-cpu`, `tqdm`, `rouge-score`, `nltk`

### Step 2: Download Dataset (~3 min)
```bash
python data/download_data.py
```
Or manually: download from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), unzip, place `twcs.csv` in `data/raw/`.

### Step 3: Process Data (~2 min)
```bash
# Set API key
set GOOGLE_API_KEY=your-key-here          # Windows
export GOOGLE_API_KEY=your-key-here       # Linux/Mac

# Run data pipeline
python src/data_pipeline.py
```
This filters for AppleSupport, builds customer→brand reply pairs, and saves to `data/processed/`.

### Step 4: Run EDA (~1 min)
```bash
python notebooks/eda.py
```
Outputs keyword/pattern analysis and recommended intents to `data/processed/eda_summary.txt`.

### Step 5: Build Golden Eval Set (~1 min)
```bash
python data/build_golden_set.py
```
Creates 200 stratified examples in `data/golden_eval_set.csv`. Labels are auto-generated starting points — see `data/golden_set_methodology.md`.

### Step 6: Run Evaluation (~5-10 min)
```bash
# Full evaluation (intent + escalation + replies + LLM judge)
python eval/run_eval.py

# Quick: baselines only (no API calls)
python eval/run_eval.py --baselines-only

# Quick: skip reply generation
python eval/run_eval.py --skip-reply-gen

# Smaller sample
python eval/run_eval.py --sample 30
```

### Step 7: Try the Agent Interactively
```bash
python src/agent.py "My iPhone won't turn on after the update"
python src/agent.py --interactive
python src/agent.py --method rule_based "How do I reset my password?"
```

## Project Structure
```
hiver/
├── data/
│   ├── download_data.py            # Dataset download script
│   ├── build_golden_set.py         # Golden eval set builder (auto-labelled)
│   ├── build_blind_golden_set.py   # Blind eval set builder (no pre-fill)
│   ├── golden_set_methodology.md   # Sampling & labelling methodology
│   ├── golden_eval_set.csv         # 200 hand-labelled examples
│   ├── blind_eval_set.csv          # 50 blind-labelled examples
│   ├── raw/                        # Raw CSV (gitignored)
│   └── processed/                  # Cleaned conversation pairs
├── src/
│   ├── data_pipeline.py            # Data cleaning & thread building
│   ├── intent_classifier.py        # 3 methods: rule-based, TF-IDF, LLM
│   ├── reply_generator.py          # RAG reply drafting (FAISS + Gemini)
│   ├── escalation.py               # Hybrid escalation decision
│   ├── product_entity.py           # Product entity extraction (POC)
│   └── agent.py                    # Main agent orchestrator + CLI
├── eval/
│   ├── run_eval.py                 # Evaluation entry point
│   ├── eval_harness.py             # Automated metrics (F1, ROUGE, BLEU)
│   ├── llm_judge.py                # LLM-as-judge with 5-dim rubric
│   └── results/
│       ├── sample_agent_outputs.md     # Raw agent output examples
│       └── sample_judge_transcripts.md # LLM judge scoring examples
├── REPORT.md                       # Full evaluation report (6 pages)
├── requirements.txt
└── README.md
```

## Key Results

| Component | Method | Headline Metric | Blind Subsample |
|-----------|--------|-----------------|-----------------|
| Intent Classification | Gemini LLM | ~78% accuracy (anchored) | **~70% accuracy (blind)** |
| Escalation | Hybrid (rules + LLM) | ~84% accuracy | ~80% accuracy |
| Reply Quality | RAG + Gemini | ~3.8/5.0 LLM judge | — |

See [REPORT.md](REPORT.md) for full results, blind validation analysis, baselines comparison, failure analysis, and the mandatory "what is misleading about my headline number?" section.

See [eval/results/](eval/results/) for raw agent outputs and LLM judge transcripts.

## Dataset
- **Primary**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, ~3M tweets)
- **Subsample used**: ~5,000 AppleSupport conversation pairs

## Citations
- Dataset: "Customer Support on Twitter" by Thought Vector (Kaggle)
- FAISS: Johnson et al., "Billion-scale similarity search with GPUs" (Facebook Research)
- RAG approach: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- LLM: Google Gemini API (gemini-2.0-flash)
- AI coding assistant used: Google Antigravity (Gemini)

## License
MIT
