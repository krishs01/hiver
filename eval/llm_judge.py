"""
llm_judge.py — LLM-as-judge evaluation for reply quality.

Uses Gemini to evaluate generated replies on a rubric:
1. Relevance (0-5): Does the reply address the customer's actual issue?
2. Tone (0-5): Is the tone empathetic, professional, and brand-appropriate?
3. Accuracy (0-5): Is the advice technically correct and safe?
4. Completeness (0-5): Does it provide actionable next steps?
5. Conciseness (0-5): Is it appropriately concise for Twitter?

Also includes human-judge agreement analysis.
"""

import os
import re
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm


JUDGE_RUBRIC = """You are an expert evaluator for customer support reply quality.
Rate the following generated reply on 5 dimensions, each from 0 (worst) to 5 (best).

RUBRIC:
1. RELEVANCE (0-5): Does the reply directly address the customer's specific issue?
   - 0: Completely off-topic
   - 3: Somewhat relevant but misses key aspects
   - 5: Directly addresses the exact issue raised

2. TONE (0-5): Is the tone empathetic, professional, and appropriate for Apple Support?
   - 0: Rude, dismissive, or robotic
   - 3: Neutral/acceptable but not empathetic
   - 5: Warm, empathetic, professional — matches Apple's brand voice

3. ACCURACY (0-5): Is the advice technically correct and safe to follow?
   - 0: Dangerously wrong advice
   - 3: Partially correct or vague
   - 5: Accurate, specific, and safe advice

4. COMPLETENESS (0-5): Does it provide clear next steps the customer can act on?
   - 0: No actionable information
   - 3: Some steps but incomplete
   - 5: Clear, complete action plan

5. CONCISENESS (0-5): Is it appropriately concise for Twitter?
   - 0: Way too long or way too short
   - 3: Acceptable length
   - 5: Perfect length — says what's needed without waste"""


def judge_single_reply(
    customer_message: str,
    generated_reply: str,
    reference_reply: str = "",
    intent: str = "",
) -> dict:
    """
    Use LLM to judge a single generated reply.
    
    Returns dict with scores for each dimension and overall score.
    """
    from google import genai
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    
    ref_section = ""
    if reference_reply:
        ref_section = f"\n[Actual brand reply for reference]: {reference_reply}"
    
    intent_section = ""
    if intent:
        intent_section = f"\n[Classified intent]: {intent}"
    
    prompt = f"""{JUDGE_RUBRIC}
{intent_section}
[Customer message]: {customer_message}

[Generated reply to evaluate]: {generated_reply}
{ref_section}

Rate the generated reply. Respond in this exact JSON format (no markdown, no code blocks):
{{"relevance": 0-5, "tone": 0-5, "accuracy": 0-5, "completeness": 0-5, "conciseness": 0-5, "overall_comment": "brief explanation"}}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    result_text = response.text.strip()
    
    # Clean markdown
    if result_text.startswith("```"):
        result_text = re.sub(r"```(?:json)?\s*", "", result_text)
        result_text = result_text.strip("`").strip()
    
    try:
        scores = json.loads(result_text)
        # Validate and clamp scores
        for dim in ["relevance", "tone", "accuracy", "completeness", "conciseness"]:
            scores[dim] = max(0, min(5, int(scores.get(dim, 3))))
        
        scores["average_score"] = np.mean([
            scores["relevance"], scores["tone"], scores["accuracy"],
            scores["completeness"], scores["conciseness"]
        ])
        
        return scores
    except (json.JSONDecodeError, ValueError):
        return {
            "relevance": 3, "tone": 3, "accuracy": 3,
            "completeness": 3, "conciseness": 3,
            "average_score": 3.0,
            "overall_comment": "Failed to parse judge response",
            "parse_error": True
        }


def judge_batch(
    eval_data: list[dict],
    delay: float = 0.5,
    verbose: bool = True,
) -> list[dict]:
    """
    Judge a batch of generated replies.
    
    Args:
        eval_data: List of dicts with keys:
            - customer_text
            - generated_reply
            - reference_reply (optional)
            - intent (optional)
        delay: Seconds between API calls (rate limiting)
        verbose: Print progress
    
    Returns:
        List of score dicts
    """
    results = []
    
    iterator = tqdm(eval_data, desc="Judging replies") if verbose else eval_data
    
    for item in iterator:
        try:
            scores = judge_single_reply(
                customer_message=item["customer_text"],
                generated_reply=item["generated_reply"],
                reference_reply=item.get("reference_reply", ""),
                intent=item.get("intent", ""),
            )
            results.append(scores)
        except Exception as e:
            results.append({
                "relevance": 0, "tone": 0, "accuracy": 0,
                "completeness": 0, "conciseness": 0,
                "average_score": 0.0,
                "overall_comment": f"Error: {str(e)}",
                "error": True
            })
        
        time.sleep(delay)
    
    return results


def compute_judge_agreement(
    human_labels: list[str],
    judge_scores: list[float],
    threshold: float = 3.5,
) -> dict:
    """
    Compute agreement between human quality labels and LLM judge scores.
    
    Human labels: 'good', 'acceptable', 'poor'
    Judge scores: average_score (0-5)
    
    We map:
        good -> judge >= 4.0
        acceptable -> 2.5 <= judge < 4.0
        poor -> judge < 2.5
    
    Then compute Cohen's kappa-like agreement.
    """
    from sklearn.metrics import cohen_kappa_score
    
    # Map human labels to numeric
    human_numeric = []
    for label in human_labels:
        if label == "good":
            human_numeric.append(2)
        elif label == "acceptable":
            human_numeric.append(1)
        else:
            human_numeric.append(0)
    
    # Map judge scores to categories
    judge_numeric = []
    for score in judge_scores:
        if score >= 4.0:
            judge_numeric.append(2)
        elif score >= 2.5:
            judge_numeric.append(1)
        else:
            judge_numeric.append(0)
    
    # Cohen's kappa
    kappa = cohen_kappa_score(human_numeric, judge_numeric)
    
    # Simple agreement rate
    exact_agreement = sum(
        1 for h, j in zip(human_numeric, judge_numeric) if h == j
    ) / len(human_numeric)
    
    # Adjacent agreement (within 1 category)
    adjacent_agreement = sum(
        1 for h, j in zip(human_numeric, judge_numeric) if abs(h - j) <= 1
    ) / len(human_numeric)
    
    return {
        "cohens_kappa": kappa,
        "exact_agreement": exact_agreement,
        "adjacent_agreement": adjacent_agreement,
        "n_samples": len(human_labels),
    }


def format_judge_results(scores: list[dict]) -> str:
    """Format aggregated judge results."""
    dims = ["relevance", "tone", "accuracy", "completeness", "conciseness", "average_score"]
    
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append("LLM JUDGE RESULTS")
    lines.append(f"{'=' * 60}")
    lines.append(f"  N evaluated: {len(scores)}")
    
    for dim in dims:
        values = [s[dim] for s in scores if dim in s]
        if values:
            lines.append(f"  {dim:15s}: {np.mean(values):.2f} ± {np.std(values):.2f} "
                        f"(min={min(values):.1f}, max={max(values):.1f})")
    
    # Score distribution
    avg_scores = [s.get("average_score", 0) for s in scores]
    lines.append(f"\n  Score distribution:")
    lines.append(f"    Excellent (≥4.0): {sum(1 for s in avg_scores if s >= 4.0)}")
    lines.append(f"    Good (3.0-3.9):   {sum(1 for s in avg_scores if 3.0 <= s < 4.0)}")
    lines.append(f"    Fair (2.0-2.9):   {sum(1 for s in avg_scores if 2.0 <= s < 3.0)}")
    lines.append(f"    Poor (<2.0):      {sum(1 for s in avg_scores if s < 2.0)}")
    
    lines.append(f"{'=' * 60}")
    
    return "\n".join(lines)
