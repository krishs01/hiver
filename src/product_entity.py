"""
product_entity.py — Lightweight product entity extraction for AppleSupport.

Extracts which Apple product(s) a customer message refers to.
Used to improve RAG retrieval precision (failure mode #5 fix).

This is a proof-of-concept — a production system would use a proper NER model.
"""

import re
from typing import List


# Product patterns — ordered by specificity (most specific first)
PRODUCT_PATTERNS = {
    "iPhone": [
        r"\biphone\s*\d*\s*(pro|max|plus|mini|se)?\b",
        r"\biphone\b",
    ],
    "iPad": [
        r"\bipad\s*(pro|air|mini)?\b",
        r"\bipad\b",
    ],
    "Mac": [
        r"\b(macbook|macbook\s*(pro|air))\b",
        r"\b(imac|mac\s*(pro|mini|studio)?)\b",
        r"\bmacos\b",
    ],
    "Apple Watch": [
        r"\b(apple\s*watch|iwatch)\b",
        r"\bwatchos\b",
    ],
    "AirPods": [
        r"\b(airpods|airpod)\s*(pro|max)?\b",
    ],
    "Apple TV": [
        r"\b(apple\s*tv|appletv)\b",
        r"\btvos\b",
    ],
    "HomePod": [
        r"\bhomepod\s*(mini)?\b",
    ],
    "iOS": [
        r"\bios\s*\d*\b",
    ],
    "iTunes": [
        r"\bitunes\b",
    ],
    "iCloud": [
        r"\bicloud\b",
    ],
    "App Store": [
        r"\bapp\s*store\b",
    ],
    "Apple Music": [
        r"\b(apple\s*music)\b",
    ],
}


def extract_products(text: str) -> list[str]:
    """
    Extract Apple product mentions from text.
    
    Returns list of product names found, ordered by position in text.
    """
    text_lower = text.lower()
    found = []
    
    for product, patterns in PRODUCT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                if product not in found:
                    found.append(product)
                break
    
    return found


def get_primary_product(text: str) -> str:
    """
    Get the primary (first mentioned) product, or 'Unknown' if none found.
    """
    products = extract_products(text)
    return products[0] if products else "Unknown"


def filter_by_product(texts_with_replies: list[dict], target_product: str) -> list[dict]:
    """
    Filter a list of conversation pairs to only those mentioning a specific product.
    Used to improve RAG retrieval precision.
    
    Args:
        texts_with_replies: List of dicts with at least 'customer_text' key
        target_product: Product to filter for
    
    Returns:
        Filtered list (may be empty if no matches)
    """
    filtered = []
    for item in texts_with_replies:
        products = extract_products(item.get("customer_text", ""))
        if target_product in products or not products:
            # Include if product matches OR if no product detected (don't exclude ambiguous)
            filtered.append(item)
    
    return filtered


if __name__ == "__main__":
    test_messages = [
        "My iPhone 13 Pro keeps crashing after the iOS 16 update",
        "AirPods won't connect to my MacBook Pro via Bluetooth",
        "How do I set up iCloud backup on my iPad?",
        "The App Store keeps crashing on my Apple Watch",
        "It's not working anymore, please help",
        "Apple Music won't play any songs on my HomePod mini",
    ]
    
    print("Product Entity Extraction (proof-of-concept)")
    print("-" * 55)
    for msg in test_messages:
        products = extract_products(msg)
        primary = get_primary_product(msg)
        print(f"  [{primary:12s}] {products} <- {msg[:50]}")
