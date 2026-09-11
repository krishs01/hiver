"""
build_blind_golden_set.py — Build a BLIND evaluation subsample.

Unlike the main golden set, this script:
- Does NOT pre-fill any labels (no rule-based seeding)
- Outputs raw tweets with empty label columns
- Intended to be labelled completely blind by a human
- Used to measure anchoring bias in the main golden set

This addresses the validity concern: if the golden set was seeded by the
same rule-based system we benchmark against, our accuracy number partly
measures "does the LLM agree with my rules" rather than "does the LLM get it right."

Usage:
    python data/build_blind_golden_set.py
    # Then manually label the output CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_blind_set(
    input_path: str = None,
    output_path: str = None,
    n_samples: int = 50,
) -> pd.DataFrame:
    """
    Build a blind evaluation set — NO pre-filled labels.
    
    Stratification is done by message length buckets (not intent)
    to avoid leaking any classification signal.
    """
    project_root = Path(__file__).resolve().parent.parent
    
    if input_path is None:
        input_path = project_root / "data" / "processed" / "apple_support_pairs.csv"
    if output_path is None:
        output_path = project_root / "data" / "blind_eval_set.csv"
    
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} conversation pairs")
    
    # Exclude examples already in the main golden set to avoid overlap
    main_golden_path = project_root / "data" / "golden_eval_set.csv"
    if main_golden_path.exists():
        main_golden = pd.read_csv(main_golden_path)
        existing_ids = set(main_golden["customer_tweet_id"].values)
        df = df[~df["customer_tweet_id"].isin(existing_ids)]
        print(f"[INFO] Excluded {len(existing_ids)} examples from main golden set")
        print(f"[INFO] Remaining pool: {len(df):,}")
    
    # Stratify by message length buckets (NOT by intent — that would leak signal)
    df = df.copy()
    df["msg_len"] = df["customer_text_clean"].fillna("").str.len()
    df["len_bucket"] = pd.cut(df["msg_len"], bins=[0, 60, 100, 150, 300], labels=["short", "medium", "long", "very_long"])
    
    # Also ensure mix of with/without prior context
    df["has_context"] = df["prior_context"].fillna("").str.strip().astype(bool)
    
    # Sample evenly from length buckets, then top up to hit exact target
    sampled = []
    n_buckets = df["len_bucket"].nunique()
    base_per_bucket = n_samples // n_buckets  # 50 // 4 = 12 each
    
    for bucket in df["len_bucket"].unique():
        bucket_df = df[df["len_bucket"] == bucket]
        n_take = min(base_per_bucket, len(bucket_df))
        s = bucket_df.sample(n=n_take, random_state=99)
        sampled.append(s)
    
    blind_df = pd.concat(sampled, ignore_index=True)
    
    # Top up from remaining pool if we're short of target
    if len(blind_df) < n_samples:
        remaining = df[~df.index.isin(blind_df.index)]
        extra = remaining.sample(n=n_samples - len(blind_df), random_state=99)
        blind_df = pd.concat([blind_df, extra], ignore_index=True)
    
    # Build output — EMPTY label columns
    output_df = pd.DataFrame({
        "example_id": range(1, len(blind_df) + 1),
        "customer_tweet_id": blind_df["customer_tweet_id"].values,
        "customer_text": blind_df["customer_text_clean"].values,
        "prior_context": blind_df["prior_context"].fillna("").values,
        "brand_reply": blind_df["brand_text_clean"].values,
        # INTENTIONALLY BLANK — to be filled by human labeller
        "true_intent": [""] * len(blind_df),
        "true_escalation": [""] * len(blind_df),
        "reply_quality_label": [""] * len(blind_df),
        "notes": [""] * len(blind_df),
    })
    
    # Shuffle
    output_df = output_df.sample(frac=1, random_state=77).reset_index(drop=True)
    output_df["example_id"] = range(1, len(output_df) + 1)
    
    output_df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Blind eval set saved to {output_path}")
    print(f"     Total examples: {len(output_df)}")
    print(f"\n     Length distribution:")
    for bucket, count in blind_df["len_bucket"].value_counts().items():
        print(f"       {bucket}: {count}")
    print(f"     With prior context: {blind_df['has_context'].sum()}")
    print(f"\n     >>> LABEL THESE COMPLETELY BLIND <<<")
    print(f"     >>> Do NOT run any classifier first <<<")
    print(f"     Valid intents: DEVICE_ISSUE, SOFTWARE_UPDATE, ACCOUNT_ACCESS,")
    print(f"       CONNECTIVITY, APP_ISSUE, BATTERY_POWER, HARDWARE_PROBLEM,")
    print(f"       HOW_TO, FEEDBACK_COMPLAINT, GENERAL_INQUIRY")
    print(f"     Valid escalation: AUTO_HANDLE, ESCALATE")
    print(f"     Valid reply quality: good, acceptable, poor")
    
    return output_df


if __name__ == "__main__":
    build_blind_set()
