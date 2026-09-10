"""
run_eval.py — Single entry point to run the full evaluation pipeline.

Evaluates the AI support agent against the golden evaluation set across:
1. Three classification methods (rule_based, tfidf, llm)
2. Escalation decisions
3. Reply quality (automated metrics + LLM judge)
4. Human-judge agreement analysis

Usage:
    python eval/run_eval.py                          # Full eval (needs API key)
    python eval/run_eval.py --baselines-only         # Rule-based + TF-IDF only
    python eval/run_eval.py --skip-reply-gen          # Skip reply generation (fast)
    python eval/run_eval.py --sample 50               # Evaluate on 50 examples

Output saved to eval/results/
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent_classifier import classify, INTENT_NAMES
from src.escalation import decide_escalation
from src.reply_generator import ReplyGenerator
from eval.eval_harness import (
    evaluate_intent, evaluate_escalation,
    evaluate_reply_quality, format_results
)
from eval.llm_judge import judge_batch, format_judge_results, compute_judge_agreement


def load_golden_set(path: str = None) -> pd.DataFrame:
    """Load the golden evaluation set."""
    if path is None:
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "data" / "golden_eval_set.csv"
    
    df = pd.read_csv(path)
    print(f"[OK] Loaded golden eval set: {len(df)} examples")
    return df


def run_intent_evaluation(
    golden_df: pd.DataFrame,
    methods: list[str] = ["rule_based", "tfidf", "llm"],
    verbose: bool = True,
) -> dict:
    """Run intent classification evaluation across methods."""
    results = {}
    
    for method in methods:
        if method == "llm" and not os.environ.get("GOOGLE_API_KEY"):
            print(f"[SKIP] {method} — no GOOGLE_API_KEY set")
            continue
        
        if verbose:
            print(f"\n[INFO] Evaluating intent classification: {method}")
        
        predictions = []
        for _, row in golden_df.iterrows():
            result = classify(
                str(row["customer_text"]),
                method=method,
                prior_context=str(row.get("prior_context", ""))
            )
            predictions.append(result["intent"])
            
            # Rate limit for LLM
            if method == "llm":
                time.sleep(0.3)
        
        true_intents = golden_df["true_intent"].tolist()
        
        eval_result = evaluate_intent(true_intents, predictions, label_names=INTENT_NAMES)
        results[method] = {
            "metrics": eval_result,
            "predictions": predictions,
        }
        
        if verbose:
            print(f"  Accuracy: {eval_result['accuracy']:.3f} | "
                  f"Macro F1: {eval_result['macro_f1']:.3f} | "
                  f"Weighted F1: {eval_result['weighted_f1']:.3f}")
    
    return results


def run_escalation_evaluation(
    golden_df: pd.DataFrame,
    intent_predictions: dict = None,
    methods: list[str] = ["rule_based", "hybrid"],
    verbose: bool = True,
) -> dict:
    """Run escalation decision evaluation."""
    results = {}
    
    for method in methods:
        if method in ("llm", "hybrid") and not os.environ.get("GOOGLE_API_KEY"):
            if method == "llm":
                print(f"[SKIP] escalation {method} — no GOOGLE_API_KEY")
                continue
            else:
                # Hybrid falls back to rules-only without API key
                pass
        
        if verbose:
            print(f"\n[INFO] Evaluating escalation: {method}")
        
        predictions = []
        for _, row in golden_df.iterrows():
            intent = ""
            if intent_predictions:
                # Use LLM predictions if available, else rule_based
                best_method = "llm" if "llm" in intent_predictions else list(intent_predictions.keys())[0]
                idx = row.name if row.name < len(intent_predictions[best_method]["predictions"]) else 0
                intent = intent_predictions[best_method]["predictions"][idx]
            
            result = decide_escalation(
                str(row["customer_text"]),
                intent=intent,
                prior_context=str(row.get("prior_context", "")),
                method=method
            )
            predictions.append("ESCALATE" if result["should_escalate"] else "AUTO_HANDLE")
            
            if method in ("llm", "hybrid"):
                time.sleep(0.3)
        
        true_escalation = golden_df["true_escalation"].tolist()
        
        eval_result = evaluate_escalation(true_escalation, predictions)
        results[method] = {
            "metrics": eval_result,
            "predictions": predictions,
        }
        
        if verbose:
            print(f"  Accuracy: {eval_result['accuracy']:.3f} | "
                  f"F1: {eval_result['f1']:.3f} | "
                  f"Precision: {eval_result['precision']:.3f} | "
                  f"Recall: {eval_result['recall']:.3f}")
    
    return results


def run_reply_evaluation(
    golden_df: pd.DataFrame,
    reply_generator: ReplyGenerator = None,
    n_samples: int = None,
    verbose: bool = True,
) -> dict:
    """Run reply generation evaluation (automated + LLM judge)."""
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("[SKIP] Reply evaluation — no GOOGLE_API_KEY")
        return {}
    
    eval_df = golden_df.copy()
    if n_samples and n_samples < len(eval_df):
        eval_df = eval_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    if verbose:
        print(f"\n[INFO] Generating replies for {len(eval_df)} examples...")
    
    # Initialize reply generator
    if reply_generator is None:
        reply_generator = ReplyGenerator()
        reply_generator.build_index(max_entries=1000)
    
    # Generate replies (RAG)
    generated_replies_rag = []
    for _, row in eval_df.iterrows():
        try:
            result = reply_generator.generate_reply(
                str(row["customer_text"]),
                intent=str(row.get("true_intent", "")),
                prior_context=str(row.get("prior_context", "")),
            )
            generated_replies_rag.append(result["reply"])
        except Exception as e:
            generated_replies_rag.append(f"[Error: {e}]")
        time.sleep(0.5)
    
    # Generate replies (no RAG baseline)
    if verbose:
        print(f"[INFO] Generating baseline replies (no RAG)...")
    
    generated_replies_no_rag = []
    for _, row in eval_df.iterrows():
        try:
            result = reply_generator.generate_reply_no_rag(
                str(row["customer_text"]),
                intent=str(row.get("true_intent", "")),
            )
            generated_replies_no_rag.append(result["reply"])
        except Exception as e:
            generated_replies_no_rag.append(f"[Error: {e}]")
        time.sleep(0.5)
    
    reference_replies = eval_df["brand_reply"].fillna("").tolist()
    
    # Automated metrics
    auto_metrics_rag = evaluate_reply_quality(reference_replies, generated_replies_rag)
    auto_metrics_no_rag = evaluate_reply_quality(reference_replies, generated_replies_no_rag)
    
    if verbose:
        print(f"  RAG ROUGE-L: {auto_metrics_rag['rouge_l_mean']:.3f} | "
              f"No-RAG ROUGE-L: {auto_metrics_no_rag['rouge_l_mean']:.3f}")
    
    # LLM Judge
    if verbose:
        print(f"\n[INFO] Running LLM judge on {len(eval_df)} replies...")
    
    judge_data_rag = [
        {
            "customer_text": str(row["customer_text"]),
            "generated_reply": reply,
            "reference_reply": str(row.get("brand_reply", "")),
            "intent": str(row.get("true_intent", "")),
        }
        for (_, row), reply in zip(eval_df.iterrows(), generated_replies_rag)
    ]
    
    judge_scores_rag = judge_batch(judge_data_rag, delay=0.5, verbose=verbose)
    
    # Human-judge agreement
    human_labels = eval_df["reply_quality_label"].fillna("acceptable").tolist()
    judge_avg_scores = [s.get("average_score", 3.0) for s in judge_scores_rag]
    
    agreement = compute_judge_agreement(human_labels, judge_avg_scores)
    
    if verbose:
        print(f"\n  Human-Judge Agreement:")
        print(f"    Cohen's κ: {agreement['cohens_kappa']:.3f}")
        print(f"    Exact agreement: {agreement['exact_agreement']:.3f}")
        print(f"    Adjacent agreement: {agreement['adjacent_agreement']:.3f}")
    
    return {
        "rag": {
            "auto_metrics": auto_metrics_rag,
            "judge_scores": judge_scores_rag,
            "generated_replies": generated_replies_rag,
        },
        "no_rag": {
            "auto_metrics": auto_metrics_no_rag,
            "generated_replies": generated_replies_no_rag,
        },
        "human_judge_agreement": agreement,
    }


def save_results(results: dict, output_dir: Path):
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = json.loads(
        json.dumps(results, default=convert)
    )
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--golden-set", type=str, default=None, help="Path to golden eval set CSV")
    parser.add_argument("--baselines-only", action="store_true", help="Only run rule_based and tfidf")
    parser.add_argument("--skip-reply-gen", action="store_true", help="Skip reply generation")
    parser.add_argument("--sample", type=int, default=None, help="Subsample size for evaluation")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "eval" / "results"
    
    # Load golden set
    golden_df = load_golden_set(args.golden_set)
    
    if args.sample and args.sample < len(golden_df):
        golden_df = golden_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(golden_df)} examples")
    
    # Determine which methods to run
    if args.baselines_only:
        intent_methods = ["rule_based", "tfidf"]
        escalation_methods = ["rule_based"]
    else:
        intent_methods = ["rule_based", "tfidf", "llm"]
        escalation_methods = ["rule_based", "hybrid"]
    
    all_results = {"timestamp": datetime.now().isoformat()}
    
    # --- Intent Classification ---
    print("\n" + "=" * 70)
    print("PHASE 1: INTENT CLASSIFICATION")
    print("=" * 70)
    
    intent_results = run_intent_evaluation(golden_df, methods=intent_methods)
    all_results["intent"] = {
        method: {"metrics": {k: v for k, v in data["metrics"].items() if k != "classification_report"}}
        for method, data in intent_results.items()
    }
    
    # Print comparison
    print("\n--- Intent Classification Comparison ---")
    print(f"{'Method':<15} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 50)
    for method, data in intent_results.items():
        m = data["metrics"]
        print(f"{method:<15} {m['accuracy']:>10.3f} {m['macro_f1']:>10.3f} {m['weighted_f1']:>12.3f}")
    
    # --- Escalation ---
    print("\n" + "=" * 70)
    print("PHASE 2: ESCALATION DECISION")
    print("=" * 70)
    
    escalation_results = run_escalation_evaluation(
        golden_df, intent_predictions=intent_results, methods=escalation_methods
    )
    all_results["escalation"] = {
        method: data["metrics"]
        for method, data in escalation_results.items()
    }
    
    # --- Reply Quality ---
    if not args.skip_reply_gen:
        print("\n" + "=" * 70)
        print("PHASE 3: REPLY QUALITY")
        print("=" * 70)
        
        reply_results = run_reply_evaluation(
            golden_df,
            n_samples=args.sample or 50,  # Default to 50 for reply eval (expensive)
        )
        
        if reply_results:
            all_results["reply_quality"] = {
                "rag_auto_metrics": reply_results["rag"]["auto_metrics"],
                "no_rag_auto_metrics": reply_results["no_rag"]["auto_metrics"],
                "human_judge_agreement": reply_results["human_judge_agreement"],
            }
            
            if reply_results.get("rag", {}).get("judge_scores"):
                print(format_judge_results(reply_results["rag"]["judge_scores"]))
    
    # --- Save ---
    save_results(all_results, output_dir)
    
    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    
    # Headline numbers
    if "llm" in intent_results:
        best = intent_results["llm"]["metrics"]
        print(f"\n🏆 Headline: Intent Accuracy = {best['accuracy']:.1%}, Macro F1 = {best['macro_f1']:.3f}")
    elif "tfidf" in intent_results:
        best = intent_results["tfidf"]["metrics"]
        print(f"\n📊 TF-IDF baseline: Intent Accuracy = {best['accuracy']:.1%}, Macro F1 = {best['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
