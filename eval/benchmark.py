"""
benchmark.py - Core benchmark runner for RootScout evaluation.

For each scenario:
  1. Generate synthetic OTLP data
  2. Build a fresh GraphBuilder and ingest the data
  3. Manually wire the graph topology (workaround for parent-span inference gap)
  4. Run ContextRetriever + RCAAgent
  5. Convert the LLM response to OpenRCA prediction format
  6. Evaluate against ground truth using evaluate.py
  7. Collect results into a CSV

Usage (via run_eval.py or directly):
    python -m eval.benchmark --scenarios all --output eval/results/run.csv
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from graph.graph_builder import GraphBuilder
from graph.context_retriever import ContextRetriever
from graph.agent import RCAAgent
from RootScout.otel_ingester import OTelIngester
from RootScout.graph_sink import GraphBuilderSink
from llm_integration.client import GeminiClient, MockClient

from eval.scenario_generator import generate_otlp
from eval.evaluate import evaluate as openrca_evaluate


# ---------------------------------------------------------------------------
# Verbose criterion logging (shared by run_scenario and run_bank_scenario)
# ---------------------------------------------------------------------------

def _log_criterion_details(details: list, predicted_component: str, predicted_dt: str, sim_threshold: float, predicted_reason: str = "") -> None:
    """Print a readable breakdown of each scored criterion."""
    if not details:
        # No parsed prediction — print raw fields so the user can see what happened
        if predicted_component:
            print(f"  Predicted component : {predicted_component}")
        if predicted_dt:
            print(f"  Predicted datetime  : {predicted_dt}")
        if predicted_reason:
            short = (predicted_reason[:120] + "...") if len(predicted_reason) > 120 else predicted_reason
            print(f"  Predicted reasoning : \"{short}\"")
        return

    scored_types = {d["type"] for d in details}

    for d in details:
        if d["type"] == "component":
            verdict = "PASS" if d["passed"] else "FAIL"
            print(f"  Predicted component : {d['predicted']}")
            print(f"  Expected component  : {d['expected']}  → {verdict}")
        elif d["type"] == "reason":
            short_pred = (d["predicted"][:120] + "...") if len(d["predicted"]) > 120 else d["predicted"]
            sim = d["similarity"]
            sim_str = f"{sim:.2f}" if sim is not None else "N/A"
            verdict = "PASS" if d["passed"] else f"FAIL (threshold {sim_threshold:.2f})"
            print(f"  Predicted reason    : \"{short_pred}\"")
            print(f"  Expected reason     : \"{d['expected']}\"")
            print(f"  Cosine similarity   : {sim_str}  → {verdict}")
        elif d["type"] == "time":
            verdict = "PASS" if d["passed"] else "FAIL"
            print(f"  Predicted datetime  : {d['predicted'] or '(none)'}  → {verdict}")
            print(f"  Expected datetime   : {d['expected']}")

    # Always show reasoning if it wasn't a scored criterion (e.g. task_1, task_3)
    if "reason" not in scored_types and predicted_reason:
        short = (predicted_reason[:120] + "...") if len(predicted_reason) > 120 else predicted_reason
        print(f"  Predicted reasoning : \"{short}\"")


# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def format_prediction(
    agent_output: Dict[str, Any],
    scenario: Dict[str, Any],
) -> str:
    """
    Convert RootScout's agent output to OpenRCA's JSON prediction string.

    RootScout returns:
        {root_cause_service, root_cause_datetime, confidence, reasoning, recommended_action}

    OpenRCA expects:
        {"root cause occurrence datetime": "...",
         "root cause component": "...",
         "root cause reason": "..."}

    NOTE: The datetime field uses the LLM's own prediction (root_cause_datetime).
    There is NO ground-truth fallback — if the model does not predict a time, the
    time criterion will score 0, which is the correct behaviour.
    """
    # Component: use root_cause_service from the LLM (fallback to raw_response parse)
    component = agent_output.get("root_cause_service", "")
    if not component:
        raw = agent_output.get("raw_response", "")
        import re
        m = re.search(r'"root_cause_service"\s*:\s*"([^"]+)"', raw)
        component = m.group(1) if m else "unknown"

    # Reason: pass the full reasoning text.
    # evaluate.py uses semantic similarity (sentence-transformers) to match this
    # against the short ground-truth phrase, so verbosity is fine.
    reason = agent_output.get("reasoning", "") or agent_output.get("raw_response", "")

    # Datetime: use the LLM's predicted fault start time.
    # Fall back to the raw response regex parse if the structured field is missing.
    # Do NOT inject ground-truth time — that would make the time criterion trivially pass.
    dt_str = agent_output.get("root_cause_datetime", "")
    if not dt_str:
        raw = agent_output.get("raw_response", "")
        import re
        m = re.search(r'"root_cause_datetime"\s*:\s*"([^"]+)"', raw)
        dt_str = m.group(1) if m else ""

    prediction = json.dumps({
        "root cause occurrence datetime": dt_str,
        "root cause component": component,
        "root cause reason": reason,
    })
    return prediction


# ---------------------------------------------------------------------------
# Graph wiring helper
# ---------------------------------------------------------------------------

def wire_graph(graph_builder: GraphBuilder, scenario: Dict[str, Any]) -> None:
    """
    Explicitly add topology edges and set node statuses.

    GraphBuilderSink's parent-service inference is heuristic-based and can
    miss edges when span attributes don't carry peer.service. This ensures
    the graph faithfully represents the scenario topology.
    """
    topology = scenario["topology"]
    fault = scenario["fault_injection"]
    root_cause_svc = fault["root_cause_service"]
    propagates_to = set(fault.get("propagates_to", []))
    error_services = propagates_to | {root_cause_svc}

    for svc in topology["services"]:
        graph_builder._ensure_node(svc)
        status = "error" if svc in error_services else "ok"
        nx.set_node_attributes(graph_builder.graph, {svc: {"status": status}})

    for src, dst in topology["edges"]:
        graph_builder._ensure_node(src)
        graph_builder._ensure_node(dst)
        graph_builder.graph.add_edge(src, dst, latency=50)

    # Attach a well-formatted error event to root-cause service.
    # graph_sink logs use {type/message} keys; agent.py prompt reads {source/kind/summary}.
    # We always add an event in the agent-compatible format so the LLM sees the error detail.
    rc_node = graph_builder.graph.nodes[root_cause_svc]
    error_msg = fault["error_message"]
    already_formatted = any(
        e.get("source") == "otel" and e.get("summary") == error_msg
        for e in rc_node.get("recent_events", [])
    )
    if not already_formatted:
        rc_node["recent_events"].append({
            "source": "otel",
            "kind": "error_log",
            "timestamp": scenario["fault_start_ts"].isoformat(),
            "summary": error_msg,
            "payload": {"error_type": fault["fault_type"]},
        })


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Dict[str, Any],
    llm_client,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single benchmark scenario end-to-end.

    Returns a result dict with: scenario_id, task_index, difficulty,
    prediction, score, passing, failing, agent_output, error (if any).
    """
    scenario_id = scenario["id"]
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Running [{scenario['task_index']}] {scenario['title']}")
        print(f"  Difficulty : {scenario['difficulty']}")
        print(f"  Expected   : {scenario['ground_truth']['root_cause_component']}")
        print(f"  Observed at: {scenario['observed_service']}")

    result: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "task_index": scenario["task_index"],
        "difficulty": scenario["difficulty"],
        "title": scenario["title"],
        "prediction": "",
        "score": 0.0,
        "passing": [],
        "failing": [],
        "agent_output": {},
        "error": None,
    }

    try:
        # 1. Generate OTLP data
        traces, metrics, logs = generate_otlp(scenario)

        # 2. Fresh graph for each scenario
        graph_builder = GraphBuilder()
        graph_sink = GraphBuilderSink(graph_builder)
        ingester = OTelIngester(sink=graph_sink)

        ingester.ingest_traces(traces)
        ingester.ingest_metrics(metrics)
        ingester.ingest_logs(logs)

        # 3. Wire explicit topology (fills gaps in heuristic parent inference)
        wire_graph(graph_builder, scenario)

        # 4. Retrieve context for observed (alerting) service
        retriever = ContextRetriever(graph_builder)
        context = retriever.get_context(scenario["observed_service"])

        if verbose:
            related = [n["service"] for n in context.get("related_nodes", [])]
            print(f"  Context    : {related}")

        # 5. Run RCA agent
        agent = RCAAgent(client=llm_client, github_output_path=None)
        agent_output = agent.analyze(context)
        result["agent_output"] = agent_output

        predicted_component = agent_output.get("root_cause_service", "")
        predicted_dt = agent_output.get("root_cause_datetime", "")
        predicted_reason = agent_output.get("reasoning", "")

        # 6. Format prediction in OpenRCA JSON format
        prediction_str = format_prediction(agent_output, scenario)
        result["prediction"] = prediction_str

        # 7. Evaluate
        scoring_points = scenario["scoring_points"]
        passing, failing, score, details = openrca_evaluate(prediction_str, scoring_points)
        result["score"] = score
        result["passing"] = passing
        result["failing"] = failing

        if verbose:
            from eval.evaluate import _SIM_THRESHOLD
            _log_criterion_details(details, predicted_component, predicted_dt, _SIM_THRESHOLD, predicted_reason)
            verdict = "PASS" if score == 1.0 else f"PARTIAL ({score:.2f})" if score > 0 else "FAIL"
            print(f"  Score      : {score:.2f} [{verdict}]")

    except Exception as e:
        result["error"] = traceback.format_exc()
        if verbose:
            print(f"  ERROR      : {e}")

    return result


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    scenarios: List[Dict[str, Any]],
    llm_client=None,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run all scenarios and optionally save results to CSV.

    Args:
        scenarios: list of scenario dicts (from scenarios.py + openrca_loader.py)
        llm_client: LLM client instance (defaults to GeminiClient, falls back to MockClient)
        output_csv: path to save predictions + scores CSV
        verbose: print progress

    Returns:
        List of result dicts
    """
    if llm_client is None:
        try:
            llm_client = GeminiClient()
            if verbose:
                print("[benchmark] Using GeminiClient")
        except Exception as e:
            if verbose:
                print(f"[benchmark] GeminiClient unavailable ({e}), using MockClient")
            llm_client = MockClient()

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, llm_client=llm_client, verbose=verbose)
        results.append(result)
        # Small sleep to avoid LLM rate limits
        time.sleep(0.5)

    _print_summary(results, verbose=verbose)

    if output_csv:
        _save_results(results, output_csv)
        if verbose:
            print(f"\n[benchmark] Results saved to {output_csv}")

    return results


# ---------------------------------------------------------------------------
# Summary + CSV helpers
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]], verbose: bool = True) -> None:
    if not verbose:
        return

    tiers = {"easy": [], "medium": [], "hard": []}
    for r in results:
        tier = r.get("difficulty", "unknown")
        tiers.setdefault(tier, []).append(r["score"])

    w = 14
    print(f"\n{'═'*56}")
    print("BENCHMARK SUMMARY")
    print(f"{'═'*56}")
    print(f"{'Class':<{w}}{'Total(#)':<{w}}{'Full pass(#)':<{w}}{'Avg score':<{w}}")
    print(f"{'─'*56}")

    grand_total = 0
    grand_correct = 0
    grand_sum = 0.0

    for tier in ["easy", "medium", "hard"]:
        scores = tiers.get(tier, [])
        total = len(scores)
        correct = sum(1 for s in scores if s == 1.0)
        avg = sum(scores) / total if total else 0
        grand_total += total
        grand_correct += correct
        grand_sum += sum(scores)
        print(f"{tier:<{w}}{total:<{w}}{correct:<{w}}{avg:.2f}")

    print(f"{'─'*56}")
    grand_avg = grand_sum / grand_total if grand_total else 0
    print(f"{'Total':<{w}}{grand_total:<{w}}{grand_correct:<{w}}{grand_avg:.2f}")
    print(f"{'═'*56}\n")


def _save_results(results: List[Dict[str, Any]], output_csv: str) -> None:
    import csv, os
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fields = ["scenario_id", "task_index", "difficulty", "title",
              "prediction", "score", "passing", "failing", "error"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Helpers to create CSVs for the file_evaluate() interface
# ---------------------------------------------------------------------------

def results_to_prediction_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Save predictions in the format expected by evaluate.file_evaluate()."""
    import csv, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prediction"])
        writer.writeheader()
        for r in results:
            writer.writerow({"prediction": r["prediction"]})


def scenarios_to_query_csv(scenarios: List[Dict[str, Any]], path: str) -> None:
    """Save scenarios as a query CSV (ground truth) for evaluate.file_evaluate()."""
    import csv, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "task_index", "scoring_points"])
        writer.writeheader()
        for s in scenarios:
            writer.writerow({
                "instruction": s["description"],
                "task_index": s["task_index"],
                "scoring_points": s["scoring_points"],
            })
