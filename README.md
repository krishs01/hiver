# AI Support Agent for @AppleSupport

An AI-powered customer support agent built on real Twitter support conversations. The agent classifies customer intents, drafts contextual replies grounded in historical data, and decides whether to auto-handle or escalate to a human.

**Brand**: AppleSupport — chosen for high volume and diverse technical support issues.

## Quick Start (< 15 minutes)

### 1. Setup
```bash
# Clone and install
git clone <repo-url> && cd hiver
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Set your Gemini API key
set GOOGLE_API_KEY=your-key-here
```

### 2. Download Data
```bash
# Option A: Kaggle API (needs kaggle.json)
python data/download_data.py

# Option B: Manual download
# Download from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
# Place twcs.csv in data/raw/
```

### 3. Process Data
```bash
python src/data_pipeline.py
```

### 4. Run the Agent
```bash
python src/agent.py "My iPhone won't turn on after the update"
```

### 5. Run Evaluation
```bash
python eval/run_eval.py
```

## Project Structure
```
hiver/
├── data/
│   ├── download_data.py          # Dataset download script
│   ├── raw/                      # Raw CSV (gitignored)
│   ├── processed/                # Cleaned conversation pairs
│   └── golden_eval_set.csv       # 200 hand-labelled examples
├── src/
│   ├── data_pipeline.py          # Data cleaning & thread building
│   ├── intent_classifier.py      # LLM-based intent classification
│   ├── reply_generator.py        # RAG reply drafting
│   ├── escalation.py             # Escalation decision logic
│   └── agent.py                  # Main agent orchestrator
├── eval/
│   ├── run_eval.py               # Evaluation entry point
│   ├── eval_harness.py           # Automated metrics
│   └── llm_judge.py              # LLM-as-judge rubric
├── notebooks/
│   └── eda.py                    # Exploratory data analysis
├── REPORT.md                     # Full evaluation report
├── requirements.txt
└── README.md
```

## Dataset
- **Primary**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~3M tweets, multi-turn threads)
- **Subsample**: ~50k AppleSupport conversations used for development

## Evaluation Summary
See [REPORT.md](REPORT.md) for full results, failure analysis, and decision log.

## License
MIT
