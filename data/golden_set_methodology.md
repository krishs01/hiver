# Golden Evaluation Set — Sampling & Labelling Methodology

## Overview
This document describes how the 200 hand-labelled examples in `golden_eval_set.csv` were created.

## Sampling Strategy

### Stratified Random Sampling
We used **stratified sampling** to ensure coverage across all intent categories and escalation scenarios:

1. **Ran the data pipeline** on the full AppleSupport subset (~5,000 conversation pairs)
2. **Applied rule-based intent classifier** to get initial intent distribution
3. **Sampled 20 examples per intent** (10 intents × 20 = 200 examples)
4. **Ensured diversity** within each intent stratum:
   - Mix of short and long messages
   - Mix of simple and complex issues
   - Mix of first-contact and multi-turn conversations
   - Include edge cases and ambiguous examples

### Deliberate Edge Case Inclusion
~15% of examples (30 examples) were deliberately chosen as **hard cases**:
- Messages that could belong to multiple intents
- Very short/vague messages
- Messages with mixed sentiment (e.g., polite but frustrated)
- Messages in unusual formats (links, emojis, abbreviations)

## Labelling Process

### Labels Applied
Each example was labelled with:
1. **`true_intent`** — One of 10 defined intents
2. **`true_escalation`** — `AUTO_HANDLE` or `ESCALATE`
3. **`reply_quality_label`** — Rating of the brand's actual reply: `good`, `acceptable`, `poor`
4. **`notes`** — Free-text observations about tricky aspects

### Labelling Guidelines
- **Intent**: Choose the *primary* intent. If genuinely ambiguous, label with the more actionable intent and note the ambiguity.
- **Escalation**: Label `ESCALATE` if a human agent would add meaningful value (safety, security, complex diagnostics, high frustration). Default to `AUTO_HANDLE` if a templated response would suffice.
- **Reply quality**: 
  - `good` = addresses the issue, correct tone, actionable
  - `acceptable` = partially addresses issue, or generic but not wrong
  - `poor` = wrong advice, ignores the problem, or robotic/unhelpful

### Quality Control
- All labels were reviewed in a second pass for consistency
- Ambiguous cases were flagged in the `notes` column
- Inter-rater-style check: re-labelled 30 random examples after a break to measure self-consistency (~90% agreement)

## Distribution Summary
| Intent | Count | Escalation Rate |
|--------|-------|-----------------|
| DEVICE_ISSUE | 20 | 15% |
| SOFTWARE_UPDATE | 20 | 10% |
| ACCOUNT_ACCESS | 20 | 40% |
| CONNECTIVITY | 20 | 10% |
| APP_ISSUE | 20 | 20% |
| BATTERY_POWER | 20 | 15% |
| HARDWARE_PROBLEM | 20 | 25% |
| HOW_TO | 20 | 5% |
| FEEDBACK_COMPLAINT | 20 | 60% |
| GENERAL_INQUIRY | 20 | 10% |
| **Total** | **200** | **~21%** |

## Limitations
- Single labeller (no formal inter-annotator agreement score)
- Self-consistency check used as proxy for reliability
- Some messages are inherently ambiguous — the "true" label is a judgment call
- Brand replies were evaluated in isolation (without full thread context in some cases)
