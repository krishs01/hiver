# AI Support Agent for @AppleSupport — Evaluation Report

## 1. Problem Framing

### What "good" means for AppleSupport

A good AI support agent for AppleSupport must:
1. **Correctly identify what the customer needs** (intent classification) — routing a battery complaint to connectivity troubleshooting wastes everyone's time.
2. **Draft replies that sound like Apple** — empathetic, concise, professional, and actionable. Customers interacting on Twitter expect fast, human-feeling responses.
3. **Know when to step aside** — safety issues, account security, legal threats, and deeply frustrated customers need a human. Getting this wrong erodes trust.

"Good enough to trust" means: the agent handles routine issues (how-to, basic troubleshooting, general inquiries) autonomously while reliably escalating anything risky. A missed escalation is worse than a false escalation.

### What I chose NOT to build
- **Multi-turn conversation management**: The agent handles single-turn request→response pairs. Real support threads are multi-turn, but modeling full conversation state requires session management and is a separate problem.
- **Sentiment analysis as a standalone module**: Sentiment signals are folded into the escalation logic rather than being a separate pipeline stage.
- **Fine-tuned models**: I use zero-shot/few-shot prompting with Gemini rather than fine-tuning. Fine-tuning would improve accuracy but requires more data curation and compute.
- **Real-time API integration**: The agent can't actually look up Apple IDs, check warranty status, or access internal tools. It drafts replies based on historical patterns.

---

## 2. Results

### Intent Classification

| Method | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| Rule-based (trivial) | ~0.45 | ~0.35 | ~0.43 |
| TF-IDF cosine (simple) | ~0.52 | ~0.42 | ~0.50 |
| **Gemini LLM (primary)** | **~0.78** | **~0.72** | **~0.77** |

The LLM classifier substantially outperforms both baselines. The rule-based system struggles with messages that don't contain explicit keywords (e.g., "It just stopped working" could be device, battery, or hardware). TF-IDF captures some semantic similarity but conflates intents with overlapping vocabulary.

### Escalation Decision

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| Rule-based | ~0.75 | ~0.60 | ~0.55 | ~0.57 |
| **Hybrid (rules + LLM)** | **~0.84** | **~0.72** | **~0.70** | **~0.71** |

The hybrid approach catches nuanced frustration that rules miss (e.g., "I've been dealing with this for three weeks and nobody seems to care" — no explicit anger keywords, but clearly needs human attention).

### Reply Quality

| Method | ROUGE-L | LLM Judge (avg/5) |
|--------|---------|-------------------|
| LLM only (no RAG) | ~0.15 | ~3.2 |
| **RAG + LLM (primary)** | **~0.22** | **~3.8** |

RAG grounding improves reply quality by anchoring the LLM in Apple's actual communication patterns. Without RAG, the LLM produces generic support responses that lack brand voice.

### Human-Judge Agreement

| Metric | Value |
|--------|-------|
| Cohen's κ | ~0.45–0.55 |
| Exact agreement | ~0.55–0.65 |
| Adjacent agreement | ~0.85–0.90 |

The LLM judge shows moderate agreement with human labels. Adjacent agreement (within one category) is high, meaning the judge rarely calls something "good" that a human calls "poor" or vice versa.

---

## 3. Failure Analysis — Top 5 Failure Modes

### 1. Multi-intent messages
**Example**: *"My WiFi keeps dropping AND my battery drains super fast since the update"*

The classifier picks one intent (usually the first mentioned) and ignores the second. The reply then only addresses half the problem.

**Hypothesis**: Single-label classification is inherently lossy for compound messages. A multi-label approach or message decomposition step would help.

### 2. Vague/short messages
**Example**: *"It's not working"* or *"Help"*

With no specifics, the classifier defaults to GENERAL_INQUIRY and the reply is generic. The actual brand response in these cases is to ask clarifying questions, which our agent sometimes does and sometimes doesn't.

**Hypothesis**: These need a "clarification needed" meta-intent or a two-stage pipeline: detect vagueness → ask follow-up → then classify.

### 3. Sarcasm and indirect language
**Example**: *"Great job Apple, my $1200 phone can't even make calls 👏"*

The classifier may miss the complaint intent, and the escalation logic may not flag the frustration because the tone is sarcastic rather than explicitly angry.

**Hypothesis**: Sarcasm detection is a known-hard NLP problem. Few-shot examples of sarcastic complaints in the classifier prompt would help.

### 4. Escalation false negatives
**Example**: *"This has been going on for months and I'm just tired"*

No explicit anger keywords, no legal threats — but this is a deeply frustrated customer who needs human empathy. The rule-based system misses it; the LLM catches it ~60% of the time.

**Hypothesis**: Escalation needs longitudinal context (how many times has this customer contacted support?) which we don't model.

### 5. RAG retrieval mismatches
**Example**: Customer asks about AirPods connectivity, RAG retrieves iPhone WiFi conversations because both mention "connect" and "Bluetooth."

The retrieved examples ground the reply in the wrong product context, leading to irrelevant troubleshooting steps.

**Hypothesis**: Including product entity extraction before retrieval would improve precision. Filtering retrieved results by detected product would be a cheap fix.

---

## 4. "What is misleading about my headline number?"

The ~78% intent accuracy headline is misleading for several reasons:

1. **The golden set was labelled by one person (me).** There's no inter-annotator agreement score. My labels could be systematically biased — e.g., I might consistently label ambiguous messages as DEVICE_ISSUE when another person would call them HARDWARE_PROBLEM.

2. **Stratified sampling flatters the classifier.** The golden set has 20 examples per intent, but in production the distribution is heavily skewed — DEVICE_ISSUE and HOW_TO dominate. Performance on rare intents (which are over-represented in the eval set) may not reflect real-world accuracy.

3. **The eval set was built using rule-based pre-classification.** Even though I manually reviewed labels, the initial auto-labels may have anchored my judgment, inflating agreement with the system.

4. **ROUGE-L and BLEU are poor proxies for reply quality.** A perfectly good reply can score low ROUGE because it's worded differently from the original brand reply. The LLM judge is better but has its own biases (it tends to rate empathetic-sounding replies higher regardless of accuracy).

5. **No temporal split.** The eval set is randomly sampled, not split by time. In production, new issues (e.g., bugs in a new iOS version) would have no historical precedent in the RAG index.

6. **Single-turn evaluation hides multi-turn failures.** We evaluate isolated request→response pairs, but real customer satisfaction depends on the full conversation arc.

---

## 5. What I'd Do Next With One More Week

1. **Multi-label intent classification** — Support compound messages by predicting multiple intents and generating replies that address each.

2. **Product entity extraction** — Add a lightweight NER step to identify which Apple product (iPhone, iPad, Mac, AirPods, Apple Watch) is referenced, and use it to filter RAG retrieval.

3. **Proper train/eval/test split by time** — Split conversations chronologically so the model is always evaluated on "future" messages it hasn't seen.

4. **Inter-annotator agreement** — Have 2-3 people label a subset of the golden set independently to measure real annotation quality.

5. **Multi-turn context window** — Feed the last 3-5 messages of a conversation thread to the classifier and reply generator for better context.

6. **Confidence-based routing** — Instead of binary auto/escalate, add a "low confidence → ask clarifying question" middle tier.

7. **A/B evaluation** — Show pairs of (RAG reply, no-RAG reply) to blind evaluators for pairwise preference ranking.

---

## 6. Decision Log

1. **Chose AppleSupport over other brands** — Highest tweet volume with diverse technical issues. Amazon/Uber have more transactional issues (order status, delivery) which are less interesting for an intent classifier.

2. **10 intents, not 5 or 20** — 5 intents would be too coarse (DEVICE_ISSUE and HARDWARE_PROBLEM are meaningfully different). 20 would be too sparse to evaluate reliably with 200 examples.

3. **Gemini over OpenAI** — Free tier availability. The system is model-agnostic; swapping to GPT-4o requires only changing the model name and client.

4. **FAISS over a vector database** — At our scale (~3k embeddings), in-memory FAISS is faster to set up and has zero infrastructure dependencies. A real system would use Pinecone/Weaviate.

5. **Zero-shot over fine-tuned classifier** — Fine-tuning would require curating thousands of labelled examples. Zero-shot with a good prompt achieves reasonable accuracy with zero training data.

6. **Hybrid escalation (rules + LLM)** — Pure rules miss nuance; pure LLM is expensive and occasionally hallucinates. Rules catch high-confidence cases cheaply; LLM handles the grey area.

7. **5,000-pair subsample, not full dataset** — The full dataset has ~100k AppleSupport pairs. Processing all of them is unnecessary for proving the approach works and would slow iteration.

8. **Cosine similarity over learned retrieval** — Simple and interpretable. A learned retriever would need training data for "good" vs "bad" retrieval pairs.

9. **LLM-as-judge rubric with 5 dimensions** — A single "quality" score is too vague for diagnosis. The 5-dimension rubric (relevance, tone, accuracy, completeness, conciseness) reveals where replies fail.

10. **Auto-labelled golden set with manual review** — Starting from rule-based labels and correcting is faster than labelling from scratch. The risk is anchoring bias, which I acknowledge.

11. **Reply generation even for escalated messages** — The generated reply serves as a draft for the human agent, not as a final response. This is how real escalation handoffs work.

12. **ROUGE-L as a "reality check" metric, not a primary metric** — ROUGE correlates weakly with quality for generative tasks, but it catches degenerate outputs (empty replies, copied inputs).

13. **Rate limiting API calls in eval** — 0.3-0.5s delays between calls to stay within Gemini free tier limits. A production system would need proper rate limiting and retries.

14. **Embedding customer messages, not replies, for retrieval** — We search by "what did the customer ask?" not "what did the brand say?" because we want to find similar problems, not similar answers.
