"""
data_pipeline.py — Load, filter, clean, and structure the Twitter support dataset.

Filters for @AppleSupport conversations, builds customer→brand reply pairs,
cleans text, and saves processed data for downstream use.

Usage:
    python src/data_pipeline.py
"""

import re
import pandas as pd
from pathlib import Path
from typing import Optional


# AppleSupport's author_id in the dataset — we identify it by finding
# the most frequent outbound (inbound=False) author. This is robust
# because AppleSupport has the highest reply volume.
BRAND_NAME = "AppleSupport"


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """Load the raw twcs.csv file."""
    print(f"[INFO] Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"[OK] Loaded {len(df):,} tweets")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def identify_brand_author(df: pd.DataFrame) -> str:
    """
    Identify AppleSupport's author_id.
    
    Strategy: The brand author is the most frequent author among outbound 
    tweets (inbound=False) whose text contains 'Apple' references.
    We look at outbound authors and pick the top one whose tweets
    reference Apple-related terms.
    """
    outbound = df[df["inbound"] == False]  # noqa: E712
    author_counts = outbound["author_id"].value_counts()
    
    # The top outbound authors are brand support accounts
    # AppleSupport is typically one of the highest-volume ones
    # We check which top author has Apple-related content
    for author_id in author_counts.head(20).index:
        sample_texts = outbound[outbound["author_id"] == author_id]["text"].head(50)
        combined = " ".join(sample_texts.fillna("").astype(str)).lower()
        if any(kw in combined for kw in ["apple", "iphone", "ipad", "mac", "ios", "dm"]):
            print(f"[OK] Identified AppleSupport author_id: {author_id}")
            print(f"[INFO] Total outbound tweets from this author: {author_counts[author_id]:,}")
            return str(author_id)
    
    # Fallback: just use the most frequent outbound author
    top_author = str(author_counts.index[0])
    print(f"[WARN] Could not confirm Apple by keyword. Using top outbound author: {top_author}")
    return top_author


def clean_text(text: str) -> str:
    """
    Clean tweet text:
    - Replace anonymized @mentions (__author_id__) with @user
    - Remove URLs
    - Normalize whitespace
    - Keep the text otherwise intact (preserve case, punctuation)
    """
    if not isinstance(text, str):
        return ""
    
    # Replace @mention placeholders like @115712 with @user
    text = re.sub(r"@\d+", "@user", text)
    
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def build_conversation_pairs(df: pd.DataFrame, brand_author_id: str) -> pd.DataFrame:
    """
    Build customer_message → brand_reply pairs.
    
    Logic:
    1. Find all inbound tweets (customer messages)
    2. For each, find the brand's response using response_tweet_id / in_response_to_tweet_id
    3. Create pairs with metadata
    """
    print("[INFO] Building conversation pairs...")
    
    # Index tweets by tweet_id for fast lookup
    tweet_lookup = df.set_index("tweet_id")
    
    # Get all brand outbound tweets
    brand_tweets = df[
        (df["author_id"].astype(str) == brand_author_id) & 
        (df["inbound"] == False)  # noqa: E712
    ]
    
    # Get all inbound (customer) tweets
    customer_tweets = df[df["inbound"] == True]  # noqa: E712
    
    pairs = []
    
    # Strategy: For each brand tweet, find the customer tweet it responds to
    for _, brand_row in brand_tweets.iterrows():
        response_to_id = brand_row.get("in_response_to_tweet_id")
        
        if pd.isna(response_to_id):
            continue
            
        try:
            response_to_id = int(float(response_to_id))
        except (ValueError, TypeError):
            continue
            
        # Look up the customer tweet
        if response_to_id in tweet_lookup.index:
            customer_row = tweet_lookup.loc[response_to_id]
            
            # Handle case where lookup returns multiple rows
            if isinstance(customer_row, pd.DataFrame):
                customer_row = customer_row.iloc[0]
            
            # Verify it's actually a customer (inbound) tweet
            if customer_row.get("inbound") != True:
                continue
                
            pairs.append({
                "customer_tweet_id": response_to_id,
                "customer_author_id": str(customer_row.get("author_id", "")),
                "customer_text": str(customer_row.get("text", "")),
                "customer_text_clean": clean_text(str(customer_row.get("text", ""))),
                "customer_created_at": customer_row.get("created_at", ""),
                "brand_tweet_id": brand_row["tweet_id"],
                "brand_text": str(brand_row.get("text", "")),
                "brand_text_clean": clean_text(str(brand_row.get("text", ""))),
                "brand_created_at": brand_row.get("created_at", ""),
            })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"[OK] Built {len(pairs_df):,} conversation pairs")
    return pairs_df


def add_conversation_context(pairs_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-turn context: check if this customer message is part of
    an ongoing thread (has a prior brand message it's responding to).
    """
    tweet_lookup = df.set_index("tweet_id")
    
    contexts = []
    for _, row in pairs_df.iterrows():
        cust_id = row["customer_tweet_id"]
        
        # Check if the customer tweet is itself a response to something
        if cust_id in tweet_lookup.index:
            cust_row = tweet_lookup.loc[cust_id]
            if isinstance(cust_row, pd.DataFrame):
                cust_row = cust_row.iloc[0]
            
            prior_id = cust_row.get("in_response_to_tweet_id")
            if pd.notna(prior_id):
                try:
                    prior_id = int(float(prior_id))
                    if prior_id in tweet_lookup.index:
                        prior_row = tweet_lookup.loc[prior_id]
                        if isinstance(prior_row, pd.DataFrame):
                            prior_row = prior_row.iloc[0]
                        contexts.append(clean_text(str(prior_row.get("text", ""))))
                        continue
                except (ValueError, TypeError):
                    pass
        
        contexts.append("")
    
    pairs_df["prior_context"] = contexts
    has_context = (pairs_df["prior_context"] != "").sum()
    print(f"[INFO] {has_context:,} pairs have prior conversation context")
    return pairs_df


def compute_basic_stats(pairs_df: pd.DataFrame) -> dict:
    """Compute basic statistics about the processed data."""
    stats = {
        "total_pairs": len(pairs_df),
        "unique_customers": pairs_df["customer_author_id"].nunique(),
        "avg_customer_msg_len": pairs_df["customer_text_clean"].str.len().mean(),
        "avg_brand_reply_len": pairs_df["brand_text_clean"].str.len().mean(),
        "median_customer_msg_len": pairs_df["customer_text_clean"].str.len().median(),
        "median_brand_reply_len": pairs_df["brand_text_clean"].str.len().median(),
        "pairs_with_context": (pairs_df["prior_context"] != "").sum(),
    }
    
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.1f}")
        else:
            print(f"  {key}: {val:,}")
    print("=" * 50)
    
    return stats


def run_pipeline(
    raw_csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_pairs: int = 5000
) -> pd.DataFrame:
    """
    Run the full data pipeline.
    
    Args:
        raw_csv_path: Path to twcs.csv. Defaults to data/raw/twcs.csv
        output_dir: Output directory. Defaults to data/processed/
        max_pairs: Maximum number of pairs to keep (for manageable subsample)
    
    Returns:
        DataFrame of conversation pairs
    """
    project_root = Path(__file__).resolve().parent.parent
    
    if raw_csv_path is None:
        raw_csv_path = project_root / "data" / "raw" / "twcs.csv"
    else:
        raw_csv_path = Path(raw_csv_path)
    
    if output_dir is None:
        output_dir = project_root / "data" / "processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df = load_raw_data(raw_csv_path)
    
    # Identify AppleSupport
    brand_author_id = identify_brand_author(df)
    
    # Filter to only tweets involving this brand
    brand_tweet_ids = set(df[df["author_id"].astype(str) == brand_author_id]["tweet_id"])
    
    # Keep tweets that are either from the brand or are responded to by the brand
    relevant_response_ids = set(
        df[df["author_id"].astype(str) == brand_author_id]["in_response_to_tweet_id"]
        .dropna()
        .astype(int)
    )
    
    relevant_df = df[
        (df["author_id"].astype(str) == brand_author_id) |
        (df["tweet_id"].isin(relevant_response_ids))
    ].copy()
    
    print(f"[INFO] Filtered to {len(relevant_df):,} tweets involving {BRAND_NAME}")
    
    # Build conversation pairs
    pairs_df = build_conversation_pairs(relevant_df, brand_author_id)
    
    # Add conversation context
    pairs_df = add_conversation_context(pairs_df, relevant_df)
    
    # Drop duplicates
    pairs_df = pairs_df.drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)
    
    # Subsample if needed
    if len(pairs_df) > max_pairs:
        print(f"[INFO] Subsampling from {len(pairs_df):,} to {max_pairs:,} pairs")
        pairs_df = pairs_df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
    
    # Compute stats
    compute_basic_stats(pairs_df)
    
    # Save
    output_path = output_dir / "apple_support_pairs.csv"
    pairs_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved processed data to {output_path}")
    
    return pairs_df


if __name__ == "__main__":
    run_pipeline()
