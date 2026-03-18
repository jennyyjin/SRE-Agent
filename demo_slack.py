#!/usr/bin/env python3
"""
Runs the full RootScout pipeline with real Slack output:
  - Ingests OTLP data 
  - Builds the service dependency graph
  - Posts a Slack incident alert when ERROR is detected
  - Runs the RCA agent 
  - Posts the structured RCA report to Slack
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
GREY = "\033[90m"
DIM = "\033[2m"

def ok(text):   print(f"  {GREEN}[ok]{RESET}  {text}")
def warn(text): print(f"  {YELLOW}[warn]{RESET} {text}")
def info(text): print(f"  {GREY}-->{RESET}  {text}")
def step(n, t): print(f"\n{BOLD}Step {n}: {t}{RESET}")
def rule():     print(f"  {DIM}{'─' * 56}{RESET}")


class DryRunNotifier:
    """Prints what would be posted to Slack without making network calls."""

    def post_incident_alert(self, service, status, signal, detail=""):
        print(f"\n  {YELLOW}[DRY-RUN] Incident Alert{RESET}")
        _print_preview({
            "service": service,
            "status": status.upper(),
            "signal": signal,
            "detail": detail,
        })

    def post_rca_report(self, focus_service, report):
        print(f"\n  {YELLOW}[DRY-RUN] RCA Report{RESET}")
        reasoning = str(report.get("reasoning", ""))
        _print_preview({
            "focus_service": focus_service,
            "root_cause": report.get("root_cause_service"),
            "confidence": f"{float(report.get('confidence', 0)) * 100:.0f}%",
            "reasoning": reasoning[:160] + ("..." if len(reasoning) > 160 else ""),
            "action": report.get("recommended_action"),
        })


def _print_preview(d):
    rule()
    for k, v in d.items():
        if v is not None and v != "":
            print(f"  {GREY}{k:<16}{RESET}{v}")
    rule()


def main():
    print(f"{BOLD}  RootScout Slack Integration Demo{RESET}")
    print(f"\n  Scenario: E-commerce checkout failures")
    print(f"  Topology: frontend -> auth-service (healthy)")
    print(f"              frontend -> cart-service (error) -> database")
    print(f"  Root cause: cart-service database connection pool exhausted")

    slack_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    dry_run = not slack_token

    print()
    if dry_run:
        warn("SLACK_BOT_TOKEN not set — running in dry-run mode")
        warn("Set SLACK_BOT_TOKEN to post real Slack messages")
    else:
        alert_ch = os.getenv("SLACK_ALERT_CHANNEL", "#incidents")
        rca_ch   = os.getenv("SLACK_RCA_CHANNEL", "") or alert_ch
        ok(f"Slack connected — alert channel: {alert_ch}, RCA channel: {rca_ch}")

    time.sleep(0.6)

    # initialize pipeline components 
    step(1, "Initialize pipeline")

    from RootScout.otel_ingester import OTelIngester
    from RootScout.graph_sink import GraphBuilderSink
    from graph.graph_builder import GraphBuilder
    from graph.context_retriever import ContextRetriever

    graph_builder = GraphBuilder()
    ok("GraphBuilder initialized")

    if dry_run:
        notifier = DryRunNotifier()
        notifier._config = type("C", (), {"alert_cooldown_seconds": 0})()
    else:
        from RootScout.slack_connector import SlackConfig, SlackNotifier
        slack_cfg = SlackConfig(
            bot_token=slack_token,
            signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
            alert_channel=os.getenv("SLACK_ALERT_CHANNEL", "#incidents"),
            rca_channel=os.getenv("SLACK_RCA_CHANNEL", ""),
            alert_cooldown_seconds=0,
        )
        notifier = SlackNotifier(slack_cfg)
    ok("SlackNotifier ready")

    from RootScout.slack_connector import SlackAlertSink
    graph_sink  = GraphBuilderSink(graph_builder)
    otel_sink   = SlackAlertSink(notifier=notifier, inner_sink=graph_sink)
    otel_ingester = OTelIngester(sink=otel_sink)
    ok("OTelIngester wired with SlackAlertSink")

    time.sleep(0.4)

    # ingest synthetic OTLP data
    step(2, "Ingest OTLP telemetry")
    info("cart-service: database query timeout after 5000 ms")

    from RootScout.test_otel_data import create_test_traces, create_test_metrics, create_test_logs
    traces  = create_test_traces()
    metrics = create_test_metrics()
    logs    = create_test_logs()

    print()
    r = otel_ingester.ingest_traces(traces)
    ok(f"Traces: {r.count} spans ingested")

    r = otel_ingester.ingest_metrics(metrics)
    ok(f"Metrics: {r.count} data points ingested")

    r = otel_ingester.ingest_logs(logs)
    ok(f"Logs: {r.count} records ingested")

    time.sleep(0.4)

    # build service dependency graph 
    step(3, "Service dependency graph")

    import networkx as nx
    for svc in ["frontend", "cart-service", "auth-service", "database"]:
        graph_builder._ensure_node(svc)
    graph_builder.graph.add_edge("frontend", "cart-service", latency=70)
    graph_builder.graph.add_edge("frontend", "auth-service", latency=50)
    graph_builder.graph.add_edge("cart-service", "database", latency=30)
    nx.set_node_attributes(graph_builder.graph, {
        "cart-service": {"status": "error"},
        "frontend": {"status": "ok"},
        "auth-service": {"status": "ok"},
        "database": {"status": "ok"},
    })
    graph_builder.graph.nodes["cart-service"]["recent_events"].append({
        "type": "error_log",
        "severity": "ERROR",
        "message": "Connection pool exhausted — timeout waiting for connection",
        "timestamp": time.time(),
    })

    print()
    g = graph_builder.graph
    order = ["frontend", "auth-service", "cart-service", "database"]
    for node in order:
        if node not in g.nodes:
            continue
        status  = g.nodes[node].get("status", "unknown")
        tag     = f"{RED}[ERROR]{RESET}" if status == "error" else f"{GREEN}[ok]   {RESET}"
        deps    = list(g.successors(node))
        dep_str = f"  ->  {', '.join(deps)}" if deps else ""
        print(f"    {tag}  {node}{dep_str}")

    time.sleep(0.4)

    # post incident alert to slack 
    step(4, "Post incident alert to Slack")
    info("cart-service ERROR detected via OTLP trace")
    print()

    notifier.post_incident_alert(
        service="cart-service",
        status="error",
        signal="trace",
        detail="Connection pool exhausted — timeout waiting for connection (5000 ms)",
    )
    ok("Incident alert sent")

    time.sleep(0.6)

    # root cause analysis
    step(5, "Root cause analysis")
    info("Extracting context from dependency graph...")

    from graph.agent import RCAAgent
    from llm_integration.client import MockClient, ClaudeClient

    context = ContextRetriever(graph_builder).get_context("cart-service")
    ok(f"Context packet: {len(context.get('related_nodes', []))} related services")

    # attach synthetic GitHub event 
    import tempfile
    gh_events = [{
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": "pull_request",
        "repo_owner": "demo-org",
        "repo_name": "ecommerce",
        "service_id": "cart-service",
        "watch_path_prefix": "services/cart",
        "pr_number": 156,
        "title": "Increase database connection pool size",
        "url": "https://github.com/demo-org/ecommerce/pull/156",
        "files": [{
            "filename": "services/cart/database.py",
            "status": "modified",
            "additions": 3,
            "deletions": 1,
            "patch": (
                "@@ -12,7 +12,9 @@ class DatabasePool:\n"
                "-        self.pool_size = 10\n"
                "+        self.pool_size = 50\n"
            ),
        }],
    }]
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for e in gh_events:
        tf.write(json.dumps(e) + "\n")
    tf.flush()
    tf_path = tf.name
    ok("GitHub context attached")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            llm = ClaudeClient()
            ok("LLM: Claude claude-sonnet-4-6")
        except Exception as exc:
            warn(f"Claude unavailable ({exc}) — falling back to mock client")
            llm = MockClient()
    else:
        llm = MockClient()
        warn("ANTHROPIC_API_KEY not set — using mock client")

    info("Sending context to LLM for analysis...")
    agent  = RCAAgent(client=llm, github_output_path=tf_path)
    report = agent.analyze(context)
    os.unlink(tf_path)

    print()
    root = report.get("root_cause_service", "unknown")
    conf = float(report.get("confidence", 0))
    why  = report.get("reasoning", "")
    fix  = report.get("recommended_action", "")

    ok(f"Root cause  : {BOLD}{root}{RESET}")
    ok(f"Confidence  : {BOLD}{conf * 100:.0f}%{RESET}")
    if why:
        ok(f"Reasoning   : {why[:120]}{'...' if len(why) > 120 else ''}")
    if fix:
        ok(f"Action      : {fix}")

    time.sleep(0.4)

    # post RCA report to slack 
    step(6, "Post RCA report to Slack")
    print()

    notifier.post_rca_report(focus_service="cart-service", report=report)
    ok("RCA report sent")

    # summary 
    if dry_run:
        print(f"\n  To post real messages to Slack, set SLACK_BOT_TOKEN in .env\n")
    else:
        print(f"\n  Alert and RCA report posted to Slack.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
