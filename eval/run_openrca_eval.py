"""
run_openrca_eval.py - Evaluate RootScout against real Bank telemetry from OpenRCA.

Unlike the synthetic benchmark (run_eval.py), this runner loads actual metric
and log CSV files from the Bank/ directory for each incident, builds a live
graph from real anomaly signals, and then runs the RCA agent.

Scoring follows the same OpenRCA methodology as the synthetic benchmark:
  component accuracy  (exact string match)
  reason accuracy     (cosine similarity ≥ 0.50)
  datetime accuracy   (within ±60 s of ground truth)

NOTE on datetime scoring
------------------------
The current RCA agent does not emit a predicted datetime in its JSON output.
For scenarios that include a time criterion (task_1, task_4, task_5, task_7),
this runner falls back to the ground-truth fault timestamp from record.csv so
those scenarios are not penalised solely for missing a datetime field.
This means time-criterion scores are artificially perfect; see README for
discussion of this limitation.

Usage
-----
# Full evaluation — 27 diverse Bank incidents with real Gemini:
    python eval/run_openrca_eval.py

# Dry-run with mock LLM (no API key needed, tests the plumbing):
    python eval/run_openrca_eval.py --mock

# Smaller quick test:
    python eval/run_openrca_eval.py --n 5

# Custom Bank data directory:
    python eval/run_openrca_eval.py --bank-dir /path/to/Bank

# Hardest scenarios only:
    python eval/run_openrca_eval.py --difficulty hard
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.openrca_bank_loader import (
    load_bank_scenarios,
    _load_windowed_metrics,
    _load_windowed_logs,
)
from eval.openrca_graph_adapter import build_bank_graph
from eval.evaluate import evaluate as openrca_evaluate, file_evaluate, report as print_report
from eval.benchmark import format_prediction, results_to_prediction_csv, scenarios_to_query_csv
from graph.context_retriever import ContextRetriever
from graph.agent import RCAAgent
from llm_integration.client import get_client, MockClient


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def run_bank_scenario(
    scenario: dict,
    llm_client,
    verbose: bool = True,
) -> dict:
    """
    End-to-end evaluation of one Bank incident.

    Steps:
      1. Load real metric + log telemetry for the ±30-min fault window
      2. Build a GraphBuilder from real anomaly signals
      3. Run ContextRetriever (BFS from apache01) + RCAAgent
      4. Score the LLM's prediction against OpenRCA ground truth
    """
    result = {
        "scenario_id":  scenario["id"],
        "task_index":   scenario["task_index"],
        "difficulty":   scenario["difficulty"],
        "title":        scenario["title"],
        "prediction":   "",
        "score":        0.0,
        "passing":      [],
        "failing":      [],
        "agent_output": {},
        "error":        None,
    }

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Running [{scenario['task_index']}] {scenario['title']}")
        print(f"  Difficulty : {scenario['difficulty']}")
        print(
            f"  Expected   : {scenario['ground_truth']['root_cause_component']}"
            f" — {scenario['ground_truth']['root_cause_reason']}"
        )

    try:
        # 1. Load real telemetry (chunked, memory-efficient)
        date_dir   = scenario["bank_date_dir"]
        load_start = scenario["bank_load_start"]
        load_end   = scenario["bank_load_end"]

        if not os.path.isdir(date_dir):
            raise FileNotFoundError(
                f"Telemetry folder not found: {date_dir}\n"
                f"  Make sure the Bank/ directory is present at the project root."
            )

        metrics_df = _load_windowed_metrics(date_dir, load_start, load_end)
        logs_df    = _load_windowed_logs(date_dir, load_start, load_end)

        if verbose:
            pods = (
                set(metrics_df["cmdb_id"].unique())
                if not metrics_df.empty else set()
            )
            print(
                f"  Telemetry  : {len(metrics_df)} metric rows, "
                f"{len(logs_df)} log rows across {len(pods)} pods"
            )

        # 2. Build graph from real data
        graph_builder = build_bank_graph(scenario, metrics_df, logs_df)

        # 3. Retrieve context for the alerting service (apache01 = web tier)
        observed_svc = scenario["observed_service"]
        if observed_svc not in graph_builder.graph:
            graph_builder._ensure_node(observed_svc)

        retriever = ContextRetriever(graph_builder)
        context   = retriever.get_context(observed_svc)

        if verbose:
            error_nodes = [
                n["service"]
                for n in context.get("related_nodes", [])
                if n.get("status") == "error"
            ]
            print(f"  Error nodes: {error_nodes if error_nodes else 'none detected'}")

        # 4. Run RCA agent (GitHub enrichment skipped — no JSONL for Bank data)
        agent = RCAAgent(client=llm_client, github_output_path=None)
        agent_output = agent.analyze(context)
        result["agent_output"] = agent_output

        predicted_component = agent_output.get("root_cause_service", "")
        predicted_dt = agent_output.get("root_cause_datetime", "")
        predicted_reason = agent_output.get("reasoning", "")

        # 5. Format prediction in OpenRCA JSON format
        prediction_str = format_prediction(agent_output, scenario)
        result["prediction"] = prediction_str

        # 6. Score against ground truth
        passing, failing, score, details = openrca_evaluate(
            prediction_str, scenario["scoring_points"]
        )
        result["score"]   = score
        result["passing"] = passing
        result["failing"] = failing

        if verbose:
            from eval.evaluate import _SIM_THRESHOLD
            from eval.benchmark import _log_criterion_details
            _log_criterion_details(details, predicted_component, predicted_dt, _SIM_THRESHOLD, predicted_reason)
            verdict = (
                "PASS"
                if score == 1.0
                else f"PARTIAL ({score:.2f})"
                if score > 0
                else "FAIL"
            )
            print(f"  Score      : {score:.2f} [{verdict}]")

    except Exception:
        result["error"] = traceback.format_exc()
        if verbose:
            print(f"  ERROR      : {result['error'].splitlines()[-1]}")

    return result


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_bank_benchmark(
    scenarios: list,
    llm_client=None,
    output_csv: str = None,
    verbose: bool = True,
) -> list:
    """Run all scenarios sequentially and return result dicts."""
    if llm_client is None:
        try:
            llm_client = get_client("gemini")
            if verbose:
                print("[bank_eval] Using GeminiClient")
        except Exception as e:
            if verbose:
                print(f"[bank_eval] GeminiClient unavailable ({e}), using MockClient")
            llm_client = MockClient()

    results = []
    for scenario in scenarios:
        result = run_bank_scenario(scenario, llm_client=llm_client, verbose=verbose)
        results.append(result)
        time.sleep(0.3)   # light buffer between LLM calls

    _print_summary(results, verbose=verbose)

    if output_csv:
        _save_results(results, output_csv)
        if verbose:
            print(f"\n[bank_eval] Full results → {output_csv}")

    return results


# ---------------------------------------------------------------------------
# Summary + CSV helpers
# ---------------------------------------------------------------------------

def _print_summary(results: list, verbose: bool = True) -> None:
    if not verbose:
        return

    tiers: dict = {}
    for r in results:
        t = r.get("difficulty", "unknown")
        tiers.setdefault(t, []).append(r["score"])

    w = 14
    print(f"\n{'═' * 56}")
    print("BANK BENCHMARK SUMMARY  (real OpenRCA telemetry)")
    print(f"{'═' * 56}")
    print(f"{'Class':<{w}}{'Total(#)':<{w}}{'Full pass(#)':<{w}}{'Avg score':<{w}}")
    print(f"{'─' * 56}")

    grand_total = grand_correct = 0
    grand_sum = 0.0
    for tier in ["easy", "medium", "hard"]:
        scores = tiers.get(tier, [])
        total   = len(scores)
        correct = sum(1 for s in scores if s == 1.0)
        avg     = sum(scores) / total if total else 0.0
        grand_total   += total
        grand_correct += correct
        grand_sum     += sum(scores)
        print(f"{tier:<{w}}{total:<{w}}{correct:<{w}}{avg:.2f}")

    print(f"{'─' * 56}")
    grand_avg = grand_sum / grand_total if grand_total else 0.0
    print(f"{'Total':<{w}}{grand_total:<{w}}{grand_correct:<{w}}{grand_avg:.2f}")
    print(f"{'═' * 56}\n")

    # Per-task breakdown
    task_scores: dict = {}
    for r in results:
        ti = r.get("task_index", "?")
        task_scores.setdefault(ti, []).append(r["score"])
    print("Per-task-type averages:")
    for ti in sorted(task_scores):
        ss = task_scores[ti]
        print(f"  {ti:<10} n={len(ss):>2}  avg={sum(ss)/len(ss):.2f}")


def _save_results(results: list, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fields = [
        "scenario_id", "task_index", "difficulty", "title",
        "prediction", "score", "passing", "failing", "error",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RootScout evaluation on real OpenRCA Bank telemetry"
    )
    parser.add_argument(
        "--bank-dir", default="Bank",
        help="Path to the Bank directory (default: Bank/)",
    )
    parser.add_argument(
        "--n", type=int, default=27,
        help="Number of scenarios to evaluate (default: 27)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockClient instead of any real LLM (fast plumbing test)",
    )
    parser.add_argument(
        "--model", default="gemini",
        help=(
            "LLM provider/model to use. Examples: gemini, claude, openai, "
            "gpt-4o, claude-sonnet, gemini-2.5-pro, openai/gpt-4o-mini "
            "(default: gemini)"
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: eval/results/bank_run_<timestamp>.csv)",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Filter scenarios by difficulty tier (default: all)",
    )
    args = parser.parse_args()

    # Load scenarios from real Bank data
    scenarios = load_bank_scenarios(data_dir=args.bank_dir, n=args.n)
    if not scenarios:
        print(
            "No scenarios loaded. Ensure the Bank/ directory contains "
            "query.csv, record.csv, and telemetry/<date>/ folders."
        )
        sys.exit(1)

    if args.difficulty != "all":
        scenarios = [s for s in scenarios if s["difficulty"] == args.difficulty]

    if not scenarios:
        print(f"No scenarios match difficulty='{args.difficulty}'. Exiting.")
        sys.exit(1)

    model_label = "MockClient" if args.mock else args.model
    print(f"\nRootScout Bank Evaluation  (real OpenRCA telemetry)")
    print(f"  Scenarios  : {len(scenarios)}")
    print(f"  Difficulty : {args.difficulty}")
    print(f"  LLM        : {model_label}")

    # LLM client
    if args.mock:
        llm_client = MockClient()
        print("  ⚠️  WARNING: Using MockClient — results are FAKE (hardcoded, not real LLM)")
    else:
        try:
            llm_client = get_client(args.model)
            print(f"  ✅  LLM client initialised: {args.model}")
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"  ⚠️  WARNING: Could not initialise '{args.model}': {e}")
            print(f"  ⚠️  FALLING BACK TO MockClient — results are FAKE")
            print(f"  ⚠️  Check your API key in .env")
            print(f"{'!'*60}\n")
            llm_client = MockClient()

    # Output paths
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = args.output or f"eval/results/bank_run_{timestamp}.csv"
    pred_csv    = results_csv.replace(".csv", "_predictions.csv")
    query_csv   = results_csv.replace(".csv", "_query.csv")
    report_csv  = results_csv.replace(".csv", "_report.csv")
    os.makedirs("eval/results", exist_ok=True)

    # Run benchmark
    results = run_bank_benchmark(
        scenarios=scenarios,
        llm_client=llm_client,
        output_csv=results_csv,
        verbose=True,
    )

    # Persist CSVs for OpenRCA-compatible file_evaluate()
    results_to_prediction_csv(results, pred_csv)
    scenarios_to_query_csv(scenarios, query_csv)

    # Final report
    print("\n--- OpenRCA-style evaluation report ---")
    try:
        file_evaluate(pred_csv, query_csv, report_csv)
        print_report(report_csv)
    except Exception as e:
        print(f"[bank_eval] Report generation failed: {e}")

    print("Artifacts written:")
    print(f"  Full results : {results_csv}")
    print(f"  Predictions  : {pred_csv}")
    print(f"  Query/GT     : {query_csv}")
    print(f"  Report       : {report_csv}")


if __name__ == "__main__":
    main()
