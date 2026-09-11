# Sample LLM Judge Transcripts

Raw examples of the LLM-as-judge evaluating generated replies.
Shows the rubric application and scoring rationale.

---

## Transcript 1: High-scoring reply

**Customer**: "My iPhone XR won't charge with any Lightning cable I try"
**Generated reply**: "That's definitely not the experience we want for you. Let's narrow this down: try charging with a different power adapter if you haven't already. Also, check the Lightning port for any debris — a toothpick can gently clear it. If neither helps, DM us your device details and we'll explore next steps."
**Reference reply**: "@user We'd like to help. Try these steps and let us know: [troubleshooting link]"

**Judge scores**:
```json
{
  "relevance": 5,
  "tone": 5,
  "accuracy": 4,
  "completeness": 5,
  "conciseness": 4,
  "average_score": 4.6,
  "overall_comment": "Directly addresses charging issue with specific, actionable troubleshooting steps. Tone is empathetic and professional. Slightly long for a single tweet but provides real value. Advice is sound — checking port debris is a common fix."
}
```

---

## Transcript 2: Mid-scoring reply

**Customer**: "How do I change my Apple ID email address?"
**Generated reply**: "Great question! You can update your Apple ID email by going to appleid.apple.com, signing in, and clicking Edit next to your email. Let us know if you need further help!"
**Reference reply**: "@user We can help with that! Send us a DM and we'll walk you through the steps."

**Judge scores**:
```json
{
  "relevance": 5,
  "tone": 4,
  "accuracy": 3,
  "completeness": 4,
  "conciseness": 5,
  "average_score": 4.2,
  "overall_comment": "Relevant and concise. The steps are approximately correct but the exact UI flow may differ from current Apple ID settings. Reference reply just redirects to DM which is less helpful. Tone is friendly but slightly less empathetic than Apple's typical voice."
}
```

---

## Transcript 3: Low-scoring reply (RAG retrieval mismatch)

**Customer**: "My AirPods keep disconnecting from my iPhone during calls"
**Generated reply**: "Sorry about that! Try going to Settings > Wi-Fi and toggle it off and back on. You can also try resetting your network settings at Settings > General > Reset > Reset Network Settings."
**Reference reply**: "@user We'd like to help! Forget the AirPods in Bluetooth settings, then re-pair them. Also check Settings > Bluetooth > [your AirPods] > i and make sure Automatic Ear Detection is on."

**Judge scores**:
```json
{
  "relevance": 2,
  "tone": 4,
  "accuracy": 1,
  "completeness": 2,
  "conciseness": 4,
  "average_score": 2.6,
  "overall_comment": "The reply gives Wi-Fi troubleshooting steps for a Bluetooth/AirPods issue — clearly retrieved the wrong historical examples. Tone is fine but the advice is irrelevant and potentially confusing. The reference reply correctly targets Bluetooth settings and AirPods-specific fixes."
}
```

**Failure mode**: RAG retrieval mismatch — both "Wi-Fi disconnecting" and "AirPods disconnecting" contain "disconnect" and "iPhone", causing the vector search to conflate them. Product-entity-filtered retrieval would fix this.

---

## Human-Judge Agreement Analysis

We compared LLM judge scores against human `reply_quality_label` annotations:

| LLM Judge Category | Human: good | Human: acceptable | Human: poor |
|---|---|---|---|
| Excellent (≥4.0) | **28** | 8 | 1 |
| Good (3.0-3.9) | 6 | **14** | 3 |
| Fair (2.0-2.9) | 1 | 5 | **7** |
| Poor (<2.0) | 0 | 1 | **3** |

**Bold** = agreement. The judge has a slight optimism bias (rates things slightly higher than human labels, especially for "acceptable" replies that get pushed to "Excellent"). This is a known issue with LLM judges — they tend to reward fluency over correctness.

**Cohen's κ ≈ 0.48** (moderate agreement)
**Adjacent agreement: 88%** (the judge rarely disagrees by more than one category)
