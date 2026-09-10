"""
intent_classifier.py — Classify customer messages into support intents.

Supports three modes:
1. rule_based: Fast keyword/regex matching (trivial baseline)
2. tfidf: TF-IDF + cosine similarity against intent descriptions (simple baseline)  
3. llm: Zero-shot/few-shot classification via Gemini API (primary system)

Intents derived from EDA of AppleSupport conversations:
    DEVICE_ISSUE, SOFTWARE_UPDATE, ACCOUNT_ACCESS, CONNECTIVITY,
    APP_ISSUE, BATTERY_POWER, HARDWARE_PROBLEM, HOW_TO,
    FEEDBACK_COMPLAINT, GENERAL_INQUIRY
"""

import os
import re
import json
from typing import Optional

# Intent definitions with descriptions and example keywords
INTENT_CATALOG = {
    "DEVICE_ISSUE": {
        "description": "Device not working, crashing, frozen, won't turn on, rebooting, bricked",
        "keywords": [
            "won't turn on", "frozen", "crash", "restart", "reboot", "boot",
            "brick", "dead", "black screen", "stuck", "not responding",
            "keeps crashing", "shut down", "won't start", "unresponsive"
        ],
        "examples": [
            "My iPhone keeps restarting on its own every few minutes",
            "iPad is completely frozen and won't respond to anything",
            "My phone just died and won't turn back on"
        ]
    },
    "SOFTWARE_UPDATE": {
        "description": "Problems related to iOS/macOS updates, upgrading, or software version issues",
        "keywords": [
            "update", "upgrade", "ios", "macos", "software", "latest version",
            "after updating", "new update", "update failed", "stuck on update",
            "downgrade", "beta", "firmware"
        ],
        "examples": [
            "After updating to iOS 16 my phone is extremely slow",
            "The update keeps failing halfway through",
            "How do I update my iPad to the latest iOS?"
        ]
    },
    "ACCOUNT_ACCESS": {
        "description": "Apple ID, iCloud, login, password, two-factor authentication, account locked",
        "keywords": [
            "apple id", "icloud", "account", "password", "sign in", "login",
            "locked out", "two factor", "2fa", "verification", "forgot password",
            "disabled", "recovery", "trust", "security"
        ],
        "examples": [
            "I'm locked out of my Apple ID and can't reset the password",
            "My iCloud storage is full but I can't manage it",
            "Two factor authentication isn't sending me the code"
        ]
    },
    "CONNECTIVITY": {
        "description": "WiFi, Bluetooth, cellular, network, signal, internet connection problems",
        "keywords": [
            "wifi", "wi-fi", "bluetooth", "cellular", "network", "connect",
            "signal", "internet", "no service", "airplane mode", "hotspot",
            "lte", "5g", "data", "paired", "disconnect"
        ],
        "examples": [
            "My iPhone keeps dropping WiFi connection every few minutes",
            "Bluetooth won't connect to my car anymore after the update",
            "No cellular service even though I have full bars"
        ]
    },
    "APP_ISSUE": {
        "description": "App Store problems, app downloads, purchases, subscriptions, app crashes",
        "keywords": [
            "app store", "app", "download", "install", "purchase",
            "subscription", "in-app", "refund", "app crash", "app won't open",
            "app update", "pending", "stuck downloading"
        ],
        "examples": [
            "I was charged for an app I didn't buy",
            "Apps keep crashing right after I open them",
            "Can't download any apps from the App Store"
        ]
    },
    "BATTERY_POWER": {
        "description": "Battery drain, charging issues, power problems, battery health",
        "keywords": [
            "battery", "drain", "charge", "charging", "power", "battery life",
            "battery health", "dies fast", "won't charge", "overheating",
            "hot", "charger", "lightning", "usb-c"
        ],
        "examples": [
            "My battery drains from 100% to 0% in 3 hours",
            "iPhone won't charge with any cable I try",
            "Phone gets extremely hot while charging"
        ]
    },
    "HARDWARE_PROBLEM": {
        "description": "Physical hardware issues - screen, speaker, camera, microphone, buttons",
        "keywords": [
            "screen", "speaker", "camera", "microphone", "headphone",
            "display", "touch", "face id", "touch id", "cracked",
            "broken", "pixel", "ghost touch", "volume", "button"
        ],
        "examples": [
            "My screen has a green line running down the side",
            "Face ID stopped working after I dropped my phone",
            "Speaker sounds distorted and crackly during calls"
        ]
    },
    "HOW_TO": {
        "description": "General how-to questions, feature usage, setup help, tips",
        "keywords": [
            "how do i", "how can i", "how to", "where do i", "where can i",
            "is there a way", "can i", "set up", "enable", "disable",
            "turn on", "turn off", "find", "change", "configure"
        ],
        "examples": [
            "How do I transfer my data to a new iPhone?",
            "Where can I find my saved passwords?",
            "How to set up parental controls on iPad?"
        ]
    },
    "FEEDBACK_COMPLAINT": {
        "description": "Customer frustration, complaints, negative feedback, dissatisfaction",
        "keywords": [
            "worst", "terrible", "horrible", "ridiculous", "unacceptable",
            "hate", "angry", "furious", "disgusted", "disappointed",
            "scam", "rip off", "waste", "garbage", "useless", "fed up"
        ],
        "examples": [
            "This is the worst customer service I've ever experienced",
            "Apple products used to be great, now they're garbage",
            "I'm so frustrated, this is the 5th time calling about the same issue"
        ]
    },
    "GENERAL_INQUIRY": {
        "description": "General questions, warranty, store info, product info, other",
        "keywords": [
            "warranty", "repair", "store", "genius bar", "appointment",
            "price", "trade in", "availability", "release", "when",
            "support", "contact", "phone number"
        ],
        "examples": [
            "Is my MacBook still under warranty?",
            "When is the new iPhone coming out?",
            "What's the trade-in value for my current phone?"
        ]
    },
}

INTENT_NAMES = list(INTENT_CATALOG.keys())


def classify_rule_based(text: str) -> str:
    """
    Trivial baseline: keyword matching.
    Returns the intent with the most keyword hits.
    """
    text_lower = text.lower()
    scores = {}
    
    for intent, info in INTENT_CATALOG.items():
        score = sum(1 for kw in info["keywords"] if kw in text_lower)
        scores[intent] = score
    
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "GENERAL_INQUIRY"
    return best


def classify_tfidf(text: str, _model_cache: dict = {}) -> str:
    """
    Simple baseline: TF-IDF cosine similarity against intent descriptions.
    Caches the vectorizer on first call.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if "vectorizer" not in _model_cache:
        # Build corpus from intent descriptions + keywords + examples
        corpus = []
        for intent, info in INTENT_CATALOG.items():
            doc = info["description"] + " " + " ".join(info["keywords"])
            doc += " " + " ".join(info["examples"])
            corpus.append(doc)
        
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        intent_vectors = vectorizer.fit_transform(corpus)
        
        _model_cache["vectorizer"] = vectorizer
        _model_cache["intent_vectors"] = intent_vectors
    
    vectorizer = _model_cache["vectorizer"]
    intent_vectors = _model_cache["intent_vectors"]
    
    text_vector = vectorizer.transform([text])
    similarities = cosine_similarity(text_vector, intent_vectors)[0]
    
    best_idx = similarities.argmax()
    return INTENT_NAMES[best_idx]


def classify_llm(text: str, prior_context: str = "") -> dict:
    """
    Primary system: LLM-based classification using Gemini.
    Returns dict with 'intent', 'confidence', and 'reasoning'.
    """
    from google import genai
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    
    intent_list = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in INTENT_CATALOG.items()
    )
    
    context_section = ""
    if prior_context and prior_context.strip():
        context_section = f"\n[Prior conversation context]: {prior_context}\n"
    
    prompt = f"""You are a customer support intent classifier for Apple Support.
Classify the following customer message into exactly ONE of these intents:

{intent_list}

{context_section}[Customer message]: {text}

Respond in this exact JSON format (no markdown, no code blocks):
{{"intent": "INTENT_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    result_text = response.text.strip()
    
    # Clean markdown code blocks if present
    if result_text.startswith("```"):
        result_text = re.sub(r"```(?:json)?\s*", "", result_text)
        result_text = result_text.strip("`").strip()
    
    try:
        result = json.loads(result_text)
        # Validate intent name
        if result.get("intent") not in INTENT_NAMES:
            result["intent"] = "GENERAL_INQUIRY"
            result["confidence"] = max(0.1, result.get("confidence", 0.5) - 0.3)
        return result
    except json.JSONDecodeError:
        # Fallback: try to extract intent name from text
        for name in INTENT_NAMES:
            if name in result_text:
                return {"intent": name, "confidence": 0.5, "reasoning": "Parsed from non-JSON response"}
        return {"intent": "GENERAL_INQUIRY", "confidence": 0.3, "reasoning": "Failed to parse LLM response"}


def classify(text: str, method: str = "llm", prior_context: str = "") -> dict:
    """
    Unified classification interface.
    
    Args:
        text: Customer message text
        method: One of 'rule_based', 'tfidf', 'llm'
        prior_context: Optional prior conversation context
    
    Returns:
        dict with 'intent', 'confidence', 'reasoning', 'method'
    """
    if method == "rule_based":
        intent = classify_rule_based(text)
        return {
            "intent": intent,
            "confidence": 0.5,
            "reasoning": "Keyword match",
            "method": "rule_based"
        }
    elif method == "tfidf":
        intent = classify_tfidf(text)
        return {
            "intent": intent,
            "confidence": 0.6,
            "reasoning": "TF-IDF similarity",
            "method": "tfidf"
        }
    elif method == "llm":
        result = classify_llm(text, prior_context)
        result["method"] = "llm"
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Quick test
    test_messages = [
        "My iPhone keeps crashing after the iOS update",
        "How do I reset my Apple ID password?",
        "Battery drains in 2 hours, this is ridiculous",
        "WiFi keeps disconnecting on my MacBook",
        "How do I transfer photos to my new phone?",
    ]
    
    print("Testing intent classifier (rule-based):")
    print("-" * 50)
    for msg in test_messages:
        result = classify(msg, method="rule_based")
        print(f"  [{result['intent']:20s}] {msg[:60]}")
    
    print("\nTesting intent classifier (TF-IDF):")
    print("-" * 50)
    for msg in test_messages:
        result = classify(msg, method="tfidf")
        print(f"  [{result['intent']:20s}] {msg[:60]}")
    
    if os.environ.get("GOOGLE_API_KEY"):
        print("\nTesting intent classifier (LLM):")
        print("-" * 50)
        for msg in test_messages:
            result = classify(msg, method="llm")
            print(f"  [{result['intent']:20s}] (conf={result['confidence']:.2f}) {msg[:60]}")
    else:
        print("\n[SKIP] LLM test — set GOOGLE_API_KEY to enable")
