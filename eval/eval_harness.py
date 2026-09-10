"""
eval_harness.py — Automated evaluation metrics for the AI support agent.

Metrics computed:
1. Intent classification: Accuracy, macro F1, per-class F1, confusion matrix
2. Escalation decision: Accuracy, precision, recall, F1
3. Reply quality: BLEU, ROUGE-L (vs. actual brand reply — rough proxy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from typing import Optional


def evaluate_intent(
    true_intents: list[str],
    predicted_intents: list[str],
    label_names: list[str] = None,
) -> dict:
    """
    Evaluate intent classification performance.
    
    Returns dict with accuracy, macro_f1, weighted_f1, per_class report, confusion matrix.
    """
    accuracy = accuracy_score(true_intents, predicted_intents)
    macro_f1 = f1_score(true_intents, predicted_intents, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_intents, predicted_intents, average="weighted", zero_division=0)
    
    report = classification_report(
        true_intents, predicted_intents,
        labels=label_names,
        zero_division=0,
        output_dict=True
    )
    
    report_str = classification_report(
        true_intents, predicted_intents,
        labels=label_names,
        zero_division=0
    )
    
    cm = confusion_matrix(true_intents, predicted_intents, labels=label_names)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "classification_report_str": report_str,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names or sorted(set(true_intents + predicted_intents)),
    }


def evaluate_escalation(
    true_escalation: list[str],
    predicted_escalation: list[str],
) -> dict:
    """
    Evaluate escalation decision performance.
    
    Treats ESCALATE as the positive class.
    """
    # Convert to binary
    true_binary = [1 if e == "ESCALATE" else 0 for e in true_escalation]
    pred_binary = [1 if e == "ESCALATE" else 0 for e in predicted_escalation]
    
    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)
    
    # Counts
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "escalation_rate_true": sum(true_binary) / len(true_binary),
        "escalation_rate_pred": sum(pred_binary) / len(pred_binary),
    }


def evaluate_reply_quality(
    reference_replies: list[str],
    generated_replies: list[str],
) -> dict:
    """
    Evaluate reply quality using automated text metrics.
    
    Note: BLEU/ROUGE against actual brand replies is a ROUGH PROXY.
    The LLM-as-judge (llm_judge.py) provides more meaningful evaluation.
    """
    from rouge_score import rouge_scorer
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    
    for ref, gen in zip(reference_replies, generated_replies):
        if not ref or not gen:
            continue
        scores = scorer.score(str(ref), str(gen))
        rouge_scores.append(scores["rougeL"].fmeasure)
    
    # Simple BLEU (unigram overlap)
    bleu_scores = []
    for ref, gen in zip(reference_replies, generated_replies):
        if not ref or not gen:
            continue
        ref_tokens = set(str(ref).lower().split())
        gen_tokens = set(str(gen).lower().split())
        if gen_tokens:
            precision = len(ref_tokens & gen_tokens) / len(gen_tokens)
            bleu_scores.append(precision)
    
    # Reply length stats
    gen_lengths = [len(str(r).split()) for r in generated_replies if r]
    ref_lengths = [len(str(r).split()) for r in reference_replies if r]
    
    return {
        "rouge_l_mean": np.mean(rouge_scores) if rouge_scores else 0.0,
        "rouge_l_std": np.std(rouge_scores) if rouge_scores else 0.0,
        "bleu_unigram_mean": np.mean(bleu_scores) if bleu_scores else 0.0,
        "avg_generated_length_words": np.mean(gen_lengths) if gen_lengths else 0.0,
        "avg_reference_length_words": np.mean(ref_lengths) if ref_lengths else 0.0,
        "n_evaluated": len(rouge_scores),
    }


def format_results(
    intent_results: dict,
    escalation_results: dict,
    reply_results: dict,
    method_name: str = "System"
) -> str:
    """Format evaluation results as a readable report."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"EVALUATION RESULTS — {method_name}")
    lines.append(f"{'=' * 70}")
    
    lines.append(f"\n📊 INTENT CLASSIFICATION")
    lines.append(f"  Accuracy:    {intent_results['accuracy']:.3f}")
    lines.append(f"  Macro F1:    {intent_results['macro_f1']:.3f}")
    lines.append(f"  Weighted F1: {intent_results['weighted_f1']:.3f}")
    lines.append(f"\n{intent_results['classification_report_str']}")
    
    lines.append(f"\n🚨 ESCALATION DECISION")
    lines.append(f"  Accuracy:    {escalation_results['accuracy']:.3f}")
    lines.append(f"  Precision:   {escalation_results['precision']:.3f}")
    lines.append(f"  Recall:      {escalation_results['recall']:.3f}")
    lines.append(f"  F1:          {escalation_results['f1']:.3f}")
    lines.append(f"  TP={escalation_results['true_positives']} FP={escalation_results['false_positives']} "
                 f"FN={escalation_results['false_negatives']} TN={escalation_results['true_negatives']}")
    
    lines.append(f"\n💬 REPLY QUALITY (vs. actual brand reply)")
    lines.append(f"  ROUGE-L:         {reply_results['rouge_l_mean']:.3f} ± {reply_results['rouge_l_std']:.3f}")
    lines.append(f"  BLEU (unigram):  {reply_results['bleu_unigram_mean']:.3f}")
    lines.append(f"  Avg gen length:  {reply_results['avg_generated_length_words']:.1f} words")
    lines.append(f"  Avg ref length:  {reply_results['avg_reference_length_words']:.1f} words")
    lines.append(f"  N evaluated:     {reply_results['n_evaluated']}")
    
    lines.append(f"\n{'=' * 70}")
    
    return "\n".join(lines)
