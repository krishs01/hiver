"""
eda.py — Exploratory Data Analysis on the AppleSupport conversation pairs.

Produces statistics and insights to inform intent definition and system design.
Outputs results to console and saves summary to data/processed/eda_summary.txt.

Usage:
    python notebooks/eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re


def load_processed_data() -> pd.DataFrame:
    """Load the processed conversation pairs."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "apple_support_pairs.csv"
    
    if not data_path.exists():
        print("[ERROR] Processed data not found. Run 'python src/data_pipeline.py' first.")
        raise FileNotFoundError(data_path)
    
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df):,} conversation pairs")
    return df


def analyze_message_lengths(df: pd.DataFrame) -> str:
    """Analyze customer and brand message length distributions."""
    output = []
    output.append("\n" + "=" * 60)
    output.append("MESSAGE LENGTH ANALYSIS")
    output.append("=" * 60)
    
    for col, label in [("customer_text_clean", "Customer"), ("brand_text_clean", "Brand")]:
        lengths = df[col].fillna("").str.len()
        word_counts = df[col].fillna("").str.split().str.len()
        
        output.append(f"\n  {label} Messages:")
        output.append(f"    Char length — mean: {lengths.mean():.0f}, median: {lengths.median():.0f}, "
                      f"min: {lengths.min()}, max: {lengths.max()}")
        output.append(f"    Word count  — mean: {word_counts.mean():.1f}, median: {word_counts.median():.0f}")
        
        # Distribution buckets
        bins = [0, 50, 100, 150, 200, 280, 500]
        labels_b = ["0-50", "51-100", "101-150", "151-200", "201-280", "280+"]
        dist = pd.cut(lengths, bins=bins, labels=labels_b).value_counts().sort_index()
        output.append(f"    Distribution: {dict(dist)}")
    
    result = "\n".join(output)
    print(result)
    return result


def analyze_common_keywords(df: pd.DataFrame) -> str:
    """Find most common keywords in customer messages to inform intent design."""
    output = []
    output.append("\n" + "=" * 60)
    output.append("KEYWORD ANALYSIS (Customer Messages)")
    output.append("=" * 60)
    
    # Common stop words to filter
    stop_words = {
        "the", "a", "an", "is", "it", "to", "and", "of", "in", "for",
        "on", "my", "i", "me", "you", "your", "this", "that", "with",
        "have", "has", "had", "be", "been", "was", "are", "not", "can",
        "but", "at", "do", "no", "so", "just", "get", "got", "how",
        "what", "when", "why", "will", "would", "could", "should",
        "from", "its", "if", "or", "as", "we", "they", "all", "up",
        "out", "about", "user", "im", "ive", "dont", "cant", "please",
        "hi", "hey", "thanks", "thank", "help", "need", "still",
        "any", "new", "one", "now", "after", "since", "every", "day",
        "time", "try", "tried", "keep", "does", "did", "am", "us",
        "been", "being", "also", "even", "going", "back", "there",
    }
    
    all_words = []
    for text in df["customer_text_clean"].fillna(""):
        words = re.findall(r"[a-z]+", text.lower())
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    word_freq = Counter(all_words)
    
    output.append("\n  Top 40 keywords:")
    for word, count in word_freq.most_common(40):
        pct = count / len(df) * 100
        output.append(f"    {word:20s} — {count:5,} ({pct:.1f}%)")
    
    result = "\n".join(output)
    print(result)
    return result


def analyze_common_patterns(df: pd.DataFrame) -> str:
    """Identify common message patterns/templates in customer messages."""
    output = []
    output.append("\n" + "=" * 60)
    output.append("PATTERN ANALYSIS (Customer Messages)")
    output.append("=" * 60)
    
    patterns = {
        "device_issue": r"(won't turn on|frozen|crash|restart|boot|brick|dead|black screen|stuck)",
        "update_problem": r"(update|upgrade|ios \d|software|latest version|after updating)",
        "battery": r"(battery|drain|charge|charging|power)",
        "connectivity": r"(wifi|wi-fi|bluetooth|cellular|network|connect|signal|internet)",
        "app_issue": r"(app store|app|apps|download|install|purchase|subscription)",
        "account": r"(apple id|icloud|account|password|sign in|login|locked out|two.factor|2fa)",
        "hardware": r"(screen|speaker|camera|microphone|headphone|display|touch|face id|touch id)",
        "performance": r"(slow|lag|hang|freeze|memory|storage|space|full)",
        "frustration": r"(worst|terrible|horrible|ridiculous|unacceptable|hate|angry|furious|disgusted|scam)",
        "how_to": r"(how do i|how can i|how to|where do i|where can i|is there a way)",
    }
    
    for name, pattern in patterns.items():
        matches = df["customer_text_clean"].fillna("").str.contains(pattern, case=False, regex=True)
        count = matches.sum()
        pct = count / len(df) * 100
        output.append(f"  {name:20s} — {count:5,} matches ({pct:.1f}%)")
    
    result = "\n".join(output)
    print(result)
    return result


def analyze_brand_reply_patterns(df: pd.DataFrame) -> str:
    """Analyze common patterns in brand replies."""
    output = []
    output.append("\n" + "=" * 60)
    output.append("BRAND REPLY PATTERNS")
    output.append("=" * 60)
    
    patterns = {
        "dm_redirect": r"(dm|direct message|send us a dm|private message)",
        "apology": r"(sorry|apologize|apologies|we understand)",
        "link_provided": r"(https?://|apple\.com|support\.apple)",
        "ask_details": r"(can you tell|which model|what version|more details|more info|let us know)",
        "troubleshoot": r"(try|restart|reset|update|check|setting|go to)",
        "escalation": r"(apple store|genius bar|contact us|call us|visit)",
    }
    
    for name, pattern in patterns.items():
        matches = df["brand_text_clean"].fillna("").str.contains(pattern, case=False, regex=True)
        count = matches.sum()
        pct = count / len(df) * 100
        output.append(f"  {name:20s} — {count:5,} matches ({pct:.1f}%)")
    
    result = "\n".join(output)
    print(result)
    return result


def analyze_sample_conversations(df: pd.DataFrame, n: int = 10) -> str:
    """Print sample conversations for qualitative review."""
    output = []
    output.append("\n" + "=" * 60)
    output.append(f"SAMPLE CONVERSATIONS (n={n})")
    output.append("=" * 60)
    
    samples = df.sample(n=min(n, len(df)), random_state=42)
    
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        output.append(f"\n--- Example {idx} ---")
        ctx = row.get("prior_context")
        if ctx and isinstance(ctx, str) and ctx.strip():
            output.append(f"  [Prior Context]: {ctx[:200]}")
        output.append(f"  [Customer]: {row['customer_text_clean'][:300]}")
        output.append(f"  [Brand]:    {row['brand_text_clean'][:300]}")
    
    result = "\n".join(output)
    print(result.encode("ascii", errors="replace").decode("ascii"))
    return result


def run_eda():
    """Run full EDA and save results."""
    df = load_processed_data()
    
    results = []
    results.append(f"EDA Summary — AppleSupport ({len(df):,} conversation pairs)")
    results.append(f"Generated by eda.py")
    results.append("")
    
    results.append(analyze_message_lengths(df))
    results.append(analyze_common_keywords(df))
    results.append(analyze_common_patterns(df))
    results.append(analyze_brand_reply_patterns(df))
    results.append(analyze_sample_conversations(df))
    
    # Intent recommendation based on patterns
    results.append("\n" + "=" * 60)
    results.append("RECOMMENDED INTENTS (based on pattern analysis)")
    results.append("=" * 60)
    intents = [
        "1. DEVICE_ISSUE      — Device not working, crashing, frozen, won't turn on",
        "2. SOFTWARE_UPDATE    — Problems after iOS/macOS update, update questions",
        "3. ACCOUNT_ACCESS     — Apple ID, iCloud, login, password, 2FA issues",
        "4. CONNECTIVITY       — WiFi, Bluetooth, cellular, network problems",
        "5. APP_ISSUE          — App Store, app downloads, purchases, subscriptions",
        "6. BATTERY_POWER      — Battery drain, charging, power issues",
        "7. HARDWARE_PROBLEM   — Screen, camera, speaker, physical damage",
        "8. HOW_TO             — General how-to questions, feature usage",
        "9. FEEDBACK_COMPLAINT — Frustration, complaints, negative feedback",
        "10. GENERAL_INQUIRY   — Other questions that don't fit above categories",
    ]
    for intent in intents:
        results.append(f"  {intent}")
    
    # Save summary
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "processed" / "eda_summary.txt"
    
    full_text = "\n".join(results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"\n[OK] EDA summary saved to {output_path}")
    return full_text


if __name__ == "__main__":
    run_eda()
