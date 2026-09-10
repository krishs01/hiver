"""
agent.py — Main AI support agent orchestrating classification, reply, and escalation.

This is the primary entry point for the support agent. It combines:
1. Intent classification (rule_based / tfidf / llm)
2. Reply generation (RAG-grounded via historical data)
3. Escalation decision (hybrid rules + LLM)

Usage:
    python src/agent.py "My iPhone won't turn on after the update"
    python src/agent.py --method llm "I've been hacked!"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from src.intent_classifier import classify
from src.escalation import decide_escalation
from src.reply_generator import ReplyGenerator


class AppleSupportAgent:
    """AI Support Agent for AppleSupport."""
    
    def __init__(
        self,
        classify_method: str = "llm",
        escalation_method: str = "hybrid",
        data_path: str = None,
        max_index_entries: int = 1000,
    ):
        """
        Initialize the support agent.
        
        Args:
            classify_method: 'rule_based', 'tfidf', or 'llm'
            escalation_method: 'rule_based', 'llm', or 'hybrid'
            data_path: Path to processed conversation pairs
            max_index_entries: Max entries for FAISS index
        """
        self.classify_method = classify_method
        self.escalation_method = escalation_method
        self.reply_generator = ReplyGenerator(data_path=data_path)
        self.max_index_entries = max_index_entries
        self._index_built = False
    
    def _ensure_index(self):
        """Lazily build the FAISS index on first use."""
        if not self._index_built:
            self.reply_generator.build_index(max_entries=self.max_index_entries)
            self._index_built = True
    
    def process_message(
        self,
        customer_message: str,
        prior_context: str = "",
    ) -> dict:
        """
        Process a customer message through the full pipeline.
        
        Args:
            customer_message: The customer's message text
            prior_context: Prior conversation context (if multi-turn)
        
        Returns:
            dict with keys:
                - intent: classified intent and metadata
                - reply: generated reply and retrieved examples
                - escalation: escalation decision and reasons
                - summary: human-readable summary
        """
        # Step 1: Classify intent
        intent_result = classify(
            customer_message,
            method=self.classify_method,
            prior_context=prior_context
        )
        
        # Step 2: Decide escalation
        escalation_result = decide_escalation(
            customer_message,
            intent=intent_result["intent"],
            prior_context=prior_context,
            method=self.escalation_method
        )
        
        # Step 3: Generate reply (even for escalated messages, for human reference)
        self._ensure_index()
        reply_result = self.reply_generator.generate_reply(
            customer_message,
            intent=intent_result["intent"],
            prior_context=prior_context,
        )
        
        # Build summary
        decision = "🔴 ESCALATE to human" if escalation_result["should_escalate"] else "🟢 AUTO-HANDLE"
        escalation_reasons = "; ".join(escalation_result.get("reasons", []))
        
        summary = (
            f"Intent: {intent_result['intent']} "
            f"(confidence: {intent_result.get('confidence', 'N/A')})\n"
            f"Decision: {decision}\n"
        )
        if escalation_result["should_escalate"]:
            summary += f"Escalation reason: {escalation_reasons}\n"
        summary += f"Suggested reply: {reply_result['reply']}"
        
        return {
            "intent": intent_result,
            "escalation": escalation_result,
            "reply": reply_result,
            "summary": summary,
        }
    
    def process_batch(
        self,
        messages: list[dict],
        verbose: bool = True
    ) -> list[dict]:
        """
        Process a batch of messages.
        
        Args:
            messages: List of dicts with 'customer_text' and optional 'prior_context'
            verbose: Print progress
        
        Returns:
            List of result dicts
        """
        results = []
        for i, msg in enumerate(messages):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(messages)}...")
            
            result = self.process_message(
                msg["customer_text"],
                msg.get("prior_context", "")
            )
            results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Apple Support AI Agent")
    parser.add_argument("message", nargs="?", help="Customer message to process")
    parser.add_argument("--method", default="llm", choices=["rule_based", "tfidf", "llm"],
                        help="Classification method")
    parser.add_argument("--escalation", default="hybrid", choices=["rule_based", "llm", "hybrid"],
                        help="Escalation method")
    parser.add_argument("--max-index", type=int, default=1000,
                        help="Max entries for FAISS index")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — keep entering messages")
    
    args = parser.parse_args()
    
    if not args.message and not args.interactive:
        parser.print_help()
        print("\nExample:")
        print('  python src/agent.py "My iPhone won\'t turn on"')
        print('  python src/agent.py --interactive')
        return
    
    # Check API key for LLM methods
    if args.method == "llm" and not os.environ.get("GOOGLE_API_KEY"):
        print("[ERROR] Set GOOGLE_API_KEY for LLM classification")
        print("  Falling back to tfidf method")
        args.method = "tfidf"
        args.escalation = "rule_based"
    
    agent = AppleSupportAgent(
        classify_method=args.method,
        escalation_method=args.escalation,
        max_index_entries=args.max_index,
    )
    
    if args.interactive:
        print("=" * 60)
        print("Apple Support AI Agent (Interactive Mode)")
        print("Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                message = input("\n📱 Customer: ").strip()
                if message.lower() in ("quit", "exit", "q"):
                    break
                if not message:
                    continue
                
                result = agent.process_message(message)
                print(f"\n{result['summary']}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] {e}")
    else:
        print(f"\n📱 Customer: {args.message}")
        print("-" * 60)
        result = agent.process_message(args.message)
        print(result["summary"])


if __name__ == "__main__":
    main()
