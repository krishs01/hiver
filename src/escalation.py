"""
escalation.py — Decide whether a customer message should be auto-handled or escalated.

Combines rule-based signals with LLM judgment for nuanced decisions.

Escalation triggers:
- Safety/legal/threat language
- Account security (hacked, unauthorized access)
- Extreme frustration (repeated contacts, strong language)
- Multi-turn unresolved issues (prior context shows ongoing problem)
- Financial disputes (refund, billing, charges)
- Complex technical issues requiring diagnostics
"""

import os
import re
import json
from typing import Optional


# Escalation signal patterns with severity weights
ESCALATION_SIGNALS = {
    "safety_legal": {
        "patterns": [
            r"\b(lawyer|attorney|sue|lawsuit|legal|court|ftc|bbb|consumer protection)\b",
            r"\b(threat|threatening|harass|discriminat)\b",
            r"\b(hurt|injur|danger|fire|smoke|burn|explod|shock)\b",
        ],
        "weight": 1.0,
        "reason": "Safety or legal concern detected"
    },
    "account_security": {
        "patterns": [
            r"\b(hack|hacked|unauthorized|stolen|phish|compromis|breach|fraud)\b",
            r"\b(someone (else|accessed|changed|logged))\b",
            r"\b(identity theft|suspicious activity)\b",
        ],
        "weight": 0.9,
        "reason": "Potential account security issue"
    },
    "extreme_frustration": {
        "patterns": [
            r"\b(worst|terrible|horrible|disgust|furious|outraged|unacceptable)\b",
            r"\b(never (buying|using|recommending))\b",
            r"(!!!|\?\?\?|[A-Z]{10,})",  # Excessive punctuation or ALL CAPS
            r"\b(been (waiting|trying|calling) for (weeks|months|hours|days))\b",
        ],
        "weight": 0.7,
        "reason": "Customer showing extreme frustration"
    },
    "financial_dispute": {
        "patterns": [
            r"\b(refund|charged|billing|overcharg|double charg|unauthorized (charge|purchase))\b",
            r"\b(money back|credit card|bank|chargeback|dispute)\b",
        ],
        "weight": 0.8,
        "reason": "Financial dispute requiring human review"
    },
    "repeated_contact": {
        "patterns": [
            r"\b(\d+\s*(th|rd|nd|st)\s*time (calling|contacting|reaching|writing))\b",
            r"\b(again|still (not|broken|having)|keeps happening|same (issue|problem))\b",
            r"\b(already (tried|called|contacted|spoke))\b",
        ],
        "weight": 0.6,
        "reason": "Customer indicates repeated unresolved contact"
    },
    "complex_technical": {
        "patterns": [
            r"\b(data loss|lost (all|my) (data|files|photos|contacts))\b",
            r"\b(backup (failed|corrupt|missing))\b",
            r"\b(enterprise|mdm|managed|deployment)\b",
        ],
        "weight": 0.7,
        "reason": "Complex technical issue requiring specialist"
    },
}


def check_escalation_rules(text: str, prior_context: str = "") -> dict:
    """
    Rule-based escalation check.
    
    Returns:
        dict with 'should_escalate', 'confidence', 'reasons', 'signals_found'
    """
    text_combined = f"{prior_context} {text}".lower()
    
    signals_found = []
    total_weight = 0.0
    reasons = []
    
    for signal_name, signal_info in ESCALATION_SIGNALS.items():
        for pattern in signal_info["patterns"]:
            if re.search(pattern, text_combined, re.IGNORECASE):
                signals_found.append(signal_name)
                total_weight += signal_info["weight"]
                reasons.append(signal_info["reason"])
                break  # One match per signal category is enough
    
    # Escalate if total weight exceeds threshold
    should_escalate = total_weight >= 0.6
    confidence = min(1.0, total_weight)
    
    return {
        "should_escalate": should_escalate,
        "confidence": confidence,
        "reasons": list(set(reasons)),
        "signals_found": signals_found,
        "method": "rule_based"
    }


def check_escalation_llm(
    text: str,
    intent: str = "",
    prior_context: str = ""
) -> dict:
    """
    LLM-based escalation decision using Gemini.
    Provides more nuanced judgment than rules alone.
    """
    from google import genai
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    
    context_section = ""
    if prior_context and prior_context.strip():
        context_section = f"\n[Prior conversation context]: {prior_context}"
    
    intent_section = ""
    if intent:
        intent_section = f"\n[Classified intent]: {intent}"
    
    prompt = f"""You are a customer support escalation decision system for Apple Support.
Determine whether this customer message should be AUTO-HANDLED by an AI agent or ESCALATED to a human agent.

ESCALATE if any of these apply:
- Safety concern (device overheating, injury, fire)
- Legal threats or regulatory complaints
- Account security issue (hacked, unauthorized access)
- Customer is extremely frustrated (repeated contacts, strong language)
- Financial dispute requiring human judgment
- Complex issue that requires diagnostics or account access
- The AI cannot resolve this without accessing internal systems

AUTO-HANDLE if:
- Standard troubleshooting question with known solution
- Simple how-to question
- General inquiry with a factual answer
- Mild frustration but straightforward issue
{context_section}{intent_section}

[Customer message]: {text}

Respond in this exact JSON format (no markdown, no code blocks):
{{"decision": "AUTO_HANDLE" or "ESCALATE", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    result_text = response.text.strip()
    
    # Clean markdown code blocks
    if result_text.startswith("```"):
        result_text = re.sub(r"```(?:json)?\s*", "", result_text)
        result_text = result_text.strip("`").strip()
    
    try:
        result = json.loads(result_text)
        return {
            "should_escalate": result.get("decision", "").upper() == "ESCALATE",
            "confidence": result.get("confidence", 0.5),
            "reasons": [result.get("reason", "LLM judgment")],
            "signals_found": [],
            "method": "llm"
        }
    except json.JSONDecodeError:
        # Fallback: check for keywords in response
        escalate = "ESCALATE" in result_text.upper()
        return {
            "should_escalate": escalate,
            "confidence": 0.4,
            "reasons": ["Parsed from non-JSON LLM response"],
            "signals_found": [],
            "method": "llm"
        }


def decide_escalation(
    text: str,
    intent: str = "",
    prior_context: str = "",
    method: str = "hybrid"
) -> dict:
    """
    Main escalation decision function.
    
    Args:
        text: Customer message
        intent: Classified intent (optional, helps LLM decision)
        prior_context: Prior conversation context
        method: 'rule_based', 'llm', or 'hybrid' (rules + LLM)
    
    Returns:
        dict with 'should_escalate', 'confidence', 'reasons', 'method'
    """
    if method == "rule_based":
        return check_escalation_rules(text, prior_context)
    
    elif method == "llm":
        return check_escalation_llm(text, intent, prior_context)
    
    elif method == "hybrid":
        # First check rules — high-confidence signals are decisive
        rule_result = check_escalation_rules(text, prior_context)
        
        if rule_result["confidence"] >= 0.9:
            # Strong rule signal — escalate without LLM
            rule_result["method"] = "hybrid (rules decisive)"
            return rule_result
        
        # Use LLM for nuanced cases
        if os.environ.get("GOOGLE_API_KEY"):
            llm_result = check_escalation_llm(text, intent, prior_context)
            
            # Combine signals
            combined_confidence = (rule_result["confidence"] * 0.4 + llm_result["confidence"] * 0.6)
            combined_reasons = list(set(rule_result["reasons"] + llm_result["reasons"]))
            
            # Either strong rules or LLM says escalate
            should_escalate = (
                rule_result["should_escalate"] or 
                llm_result["should_escalate"]
            )
            
            return {
                "should_escalate": should_escalate,
                "confidence": combined_confidence,
                "reasons": combined_reasons if combined_reasons else ["No escalation needed"],
                "signals_found": rule_result["signals_found"],
                "method": "hybrid"
            }
        else:
            rule_result["method"] = "hybrid (rules only, no API key)"
            return rule_result
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    test_cases = [
        ("My iPhone won't connect to WiFi", "CONNECTIVITY"),
        ("I've been hacked! Someone changed my Apple ID password!", "ACCOUNT_ACCESS"),
        ("This is TERRIBLE, 5th time calling, I want a refund NOW!!!", "FEEDBACK_COMPLAINT"),
        ("How do I set up iCloud backup?", "HOW_TO"),
        ("My phone exploded while charging and burned my desk", "HARDWARE_PROBLEM"),
        ("I was charged $99 twice for the same subscription", "FINANCIAL_DISPUTE"),
    ]
    
    print("Testing escalation (rule-based):")
    print("-" * 60)
    for msg, intent in test_cases:
        result = decide_escalation(msg, intent=intent, method="rule_based")
        decision = "🔴 ESCALATE" if result["should_escalate"] else "🟢 AUTO"
        print(f"  {decision} (conf={result['confidence']:.2f}) {msg[:55]}")
        if result["reasons"]:
            print(f"         Reasons: {'; '.join(result['reasons'][:2])}")
