"""
build_golden_set.py — Build the golden evaluation set from processed data.

This script:
1. Loads processed AppleSupport conversation pairs
2. Applies rule-based classification for initial stratification
3. Samples ~20 per intent with diversity
4. Outputs a CSV for manual labelling/review

The output CSV must be MANUALLY REVIEWED AND CORRECTED — the automated
labels are starting points, not ground truth.

Usage:
    python data/build_golden_set.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.intent_classifier import classify_rule_based, INTENT_NAMES
from src.escalation import check_escalation_rules


def build_golden_set(
    input_path: str = None,
    output_path: str = None,
    samples_per_intent: int = 20
) -> pd.DataFrame:
    """Build stratified golden evaluation set."""
    
    project_root = Path(__file__).resolve().parent.parent
    
    if input_path is None:
        input_path = project_root / "data" / "processed" / "apple_support_pairs.csv"
    if output_path is None:
        output_path = project_root / "data" / "golden_eval_set.csv"
    
    # Load processed data
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} conversation pairs")
    
    # Apply rule-based classification for stratification
    print("[INFO] Applying rule-based classification for stratification...")
    df["auto_intent"] = df["customer_text_clean"].fillna("").apply(classify_rule_based)
    
    # Apply rule-based escalation
    def get_escalation(row):
        result = check_escalation_rules(
            str(row.get("customer_text_clean", "")),
            str(row.get("prior_context", ""))
        )
        return "ESCALATE" if result["should_escalate"] else "AUTO_HANDLE"
    
    df["auto_escalation"] = df.apply(get_escalation, axis=1)
    
    # Rate brand reply quality heuristically (starting point for manual review)
    def auto_rate_reply(row):
        reply = str(row.get("brand_text_clean", "")).lower()
        customer = str(row.get("customer_text_clean", "")).lower()
        
        # Poor: very short/generic
        if len(reply) < 30:
            return "poor"
        
        # Good indicators: asks specific questions, provides steps, empathetic
        good_signals = ["try", "go to", "settings", "let us know", "sorry", "understand"]
        good_count = sum(1 for s in good_signals if s in reply)
        
        if good_count >= 3:
            return "good"
        elif good_count >= 1:
            return "acceptable"
        else:
            return "acceptable"
    
    df["auto_reply_quality"] = df.apply(auto_rate_reply, axis=1)
    
    # Stratified sampling
    print(f"[INFO] Sampling {samples_per_intent} per intent...")
    golden_rows = []
    
    for intent in INTENT_NAMES:
        intent_df = df[df["auto_intent"] == intent]
        
        if len(intent_df) == 0:
            print(f"  [WARN] No examples for {intent}, will fill from GENERAL_INQUIRY")
            continue
        
        n_sample = min(samples_per_intent, len(intent_df))
        
        # Try to get diverse samples: mix of message lengths
        if len(intent_df) >= n_sample * 2:
            # Sort by message length and pick evenly spaced samples
            intent_df = intent_df.copy()
            intent_df["msg_len"] = intent_df["customer_text_clean"].fillna("").str.len()
            intent_df = intent_df.sort_values("msg_len")
            
            # Pick evenly spaced indices
            indices = np.linspace(0, len(intent_df) - 1, n_sample, dtype=int)
            sampled = intent_df.iloc[indices]
        else:
            sampled = intent_df.sample(n=n_sample, random_state=42)
        
        golden_rows.append(sampled)
        print(f"  {intent:20s}: {len(sampled)} examples sampled from {len(intent_df)} available")
    
    golden_df = pd.concat(golden_rows, ignore_index=True)
    
    # Build output format
    output_df = pd.DataFrame({
        "example_id": range(1, len(golden_df) + 1),
        "customer_tweet_id": golden_df["customer_tweet_id"].values,
        "customer_text": golden_df["customer_text_clean"].values,
        "prior_context": golden_df["prior_context"].fillna("").values,
        "true_intent": golden_df["auto_intent"].values,  # MUST BE MANUALLY REVIEWED
        "true_escalation": golden_df["auto_escalation"].values,  # MUST BE MANUALLY REVIEWED
        "brand_reply": golden_df["brand_text_clean"].values,
        "reply_quality_label": golden_df["auto_reply_quality"].values,  # MUST BE MANUALLY REVIEWED
        "notes": [""] * len(golden_df),
    })
    
    # Shuffle so intents aren't grouped (reduces labelling bias)
    output_df = output_df.sample(frac=1, random_state=123).reset_index(drop=True)
    output_df["example_id"] = range(1, len(output_df) + 1)
    
    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\n[OK] Golden eval set saved to {output_path}")
    print(f"     Total examples: {len(output_df)}")
    print(f"\n⚠️  IMPORTANT: The labels are AUTO-GENERATED starting points.")
    print(f"     You MUST manually review and correct them before evaluation.")
    
    # Print distribution
    print(f"\n  Intent distribution:")
    for intent, count in output_df["true_intent"].value_counts().items():
        print(f"    {intent:20s}: {count}")
    
    print(f"\n  Escalation distribution:")
    for esc, count in output_df["true_escalation"].value_counts().items():
        print(f"    {esc:15s}: {count}")
    
    return output_df


if __name__ == "__main__":
    build_golden_set()
