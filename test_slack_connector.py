#!/usr/bin/env python3
"""
Tests for the Slack connector 
"""

import sys
import os
import json
import time
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
INFO = "\033[34m INFO\033[0m"

def check(label: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    msg = f"[{tag}] {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    if not condition:
        sys.exit(1)


# Test SlackConfig and slack_config_from_env
def test_config():
    print("\n── Test: SlackConfig ──────────────────────────────────────────")
    from RootScout.slack_connector import SlackConfig, slack_config_from_env

    # no token -> returns None
    os.environ.pop("SLACK_BOT_TOKEN", None)
    check("slack_config_from_env returns None when token missing",
          slack_config_from_env() is None)

    # with token -> returns config
    os.environ["SLACK_BOT_TOKEN"] = "xoxb-test-token"
    os.environ["SLACK_ALERT_CHANNEL"] = "#test-incidents"
    os.environ["SLACK_ALERT_COOLDOWN_SECONDS"] = "60"
    cfg = slack_config_from_env()
    check("slack_config_from_env returns SlackConfig when token present", cfg is not None)
    check("alert_channel read from env", cfg.alert_channel == "#test-incidents")
    check("cooldown read from env", cfg.alert_cooldown_seconds == 60)

    os.environ.pop("SLACK_BOT_TOKEN", None)
    os.environ.pop("SLACK_ALERT_CHANNEL", None)
    os.environ.pop("SLACK_ALERT_COOLDOWN_SECONDS", None)
    print()


# Test SlackNotifier block generation (no network call)
def test_block_generation():
    print("── Test: Block Kit message formatting ─────────────────────────")
    from RootScout.slack_connector import SlackConfig, SlackNotifier

    cfg = SlackConfig(bot_token="xoxb-dummy", alert_channel="#incidents")
    notifier = SlackNotifier(cfg)

    # alert blocks
    blocks = notifier._build_alert_blocks(
        service="cart-service", status="error", signal="trace",
        detail="Database timeout after 5000ms"
    )
    check("Alert blocks is a non-empty list", isinstance(blocks, list) and len(blocks) > 0)
    check("Alert block has a header", blocks[0]["type"] == "header")
    section = next((b for b in blocks if b["type"] == "section"), None)
    check("Alert block has section with service name",
          section is not None and
          any("cart-service" in f.get("text", "") for f in section.get("fields", [])))

    # RCA report blocks
    report = {
        "root_cause_service": "cart-service",
        "confidence": 0.95,
        "reasoning": "The frontend alert is downstream of cart-service failure.",
        "recommended_action": "kubectl rollout undo deployment/cart-service",
    }
    rca_blocks = notifier._build_rca_blocks("frontend", report)
    check("RCA blocks is a non-empty list", isinstance(rca_blocks, list) and len(rca_blocks) > 0)
    check("RCA block has a header", rca_blocks[0]["type"] == "header")

    # find the section that shows root cause
    fields_text = " ".join(
        f.get("text", "")
        for b in rca_blocks if b.get("type") == "section"
        for f in b.get("fields", [])
    )
    check("RCA block shows root cause service", "cart-service" in fields_text)
    check("RCA block shows confidence 95%", "95%" in fields_text)

    action_block = next(
        (b for b in rca_blocks
         if b.get("type") == "section" and
         "kubectl rollout undo" in json.dumps(b)),
        None
    )
    check("RCA block includes recommended action", action_block is not None)
    print()


# Test SlackAlertSink – deduplication and forwarding
def test_alert_sink():
    print("── Test: SlackAlertSink ────────────────────────────────────────")
    from RootScout.slack_connector import SlackConfig, SlackNotifier, SlackAlertSink
    from RootScout.otel_ingester import TelemetrySink

    posted_alerts = []

    class FakeNotifier:
        """Notifier that records calls instead of hitting Slack."""
        def __init__(self):
            self._config = SlackConfig(
                bot_token="dummy", alert_cooldown_seconds=5
            )
        def post_incident_alert(self, service, status, signal, detail=""):
            posted_alerts.append({"service": service, "status": status})

    class RecordingSink(TelemetrySink):
        def __init__(self):
            self.records = []
        def emit(self, record):
            self.records.append(record)

    inner = RecordingSink()
    sink = SlackAlertSink(notifier=FakeNotifier(), inner_sink=inner)

    # OK record -> no alert
    sink.emit({"signal": "trace", "service": "auth-service", "status_code": 1})
    check("OK record does not trigger alert", len(posted_alerts) == 0)
    check("OK record is forwarded to inner sink", len(inner.records) == 1)

    # ERROR record -> alert fires
    sink.emit({"signal": "trace", "service": "cart-service", "status_code": 2,
               "name": "GET /cart/items"})
    check("ERROR record triggers alert", len(posted_alerts) == 1)
    check("Alert carries correct service name", posted_alerts[0]["service"] == "cart-service")

    # Second ERROR within cooldown -> suppressed
    sink.emit({"signal": "trace", "service": "cart-service", "status_code": 2})
    check("Duplicate ERROR within cooldown is suppressed", len(posted_alerts) == 1)

    # Different service -> alert fires immediately
    sink.emit({"signal": "trace", "service": "database", "status_code": 2})
    check("ERROR for different service fires immediately", len(posted_alerts) == 2)

    # status="error" string form also triggers
    sink2 = SlackAlertSink(notifier=FakeNotifier())
    sink2.emit({"signal": "log", "service": "frontend", "status": "error"})
    check("status='error' string form also triggers alert", len(posted_alerts) == 3)
    print()


# Test SlackCommandHandler – signature verification
def test_signature_verification():
    print("── Test: Slack request signature verification ──────────────────")
    import hashlib, hmac as hmac_mod, time as time_mod
    from RootScout.slack_connector import SlackConfig, SlackCommandHandler

    secret = "my-signing-secret"
    cfg = SlackConfig(bot_token="xoxb-dummy", signing_secret=secret)
    handler = SlackCommandHandler(cfg)

    body = b"command=%2Frca&text=cart-service&response_url=https%3A%2F%2Fhooks.slack.com%2F..."
    timestamp = str(int(time_mod.time()))
    sig_base = f"v0:{timestamp}:{body.decode()}"
    good_sig = "v0=" + hmac_mod.new(
        secret.encode(), sig_base.encode(), hashlib.sha256
    ).hexdigest()

    check("Valid signature accepted", handler.verify_signature(body, timestamp, good_sig))
    check("Tampered signature rejected",
          not handler.verify_signature(body, timestamp, good_sig[:-4] + "0000"))
    check("Stale timestamp rejected",
          not handler.verify_signature(body, str(int(time_mod.time()) - 400), good_sig))

    cfg_no_secret = SlackConfig(bot_token="xoxb-dummy", signing_secret="")
    handler_no_secret = SlackCommandHandler(cfg_no_secret)
    check("Empty signing secret skips verification",
          handler_no_secret.verify_signature(body, "bad-ts", "bad-sig"))
    print()


# Test full pipeline integration 
def test_pipeline_integration():
    print("── Test: Full pipeline (dry-run, synthetic data) ────────────────")

    from RootScout.slack_connector import SlackConfig, SlackNotifier, SlackAlertSink
    from RootScout.otel_ingester import OTelIngester
    from RootScout.graph_sink import GraphBuilderSink, ComposedSink
    from RootScout.test_otel_data import create_test_traces, create_test_metrics, create_test_logs
    from graph.graph_builder import GraphBuilder

    posted = {"alerts": [], "reports": []}

    class CapturingNotifier:
        _config = SlackConfig(bot_token="dummy", alert_cooldown_seconds=0)
        def post_incident_alert(self, service, status, signal, detail=""):
            posted["alerts"].append(service)
            print(f"  [Slack alert captured] {service} -> {status}")
        def post_rca_report(self, focus_service, report):
            posted["reports"].append(focus_service)
            print(f"  [Slack RCA captured] focus={focus_service} "
                  f"root_cause={report.get('root_cause_service')} "
                  f"confidence={report.get('confidence')}")

    # build the pipeline: OTel -> Graph + SlackAlertSink
    graph_builder = GraphBuilder()
    graph_sink = GraphBuilderSink(graph_builder)
    base_sink = ComposedSink(graph_sink)
    otel_sink = SlackAlertSink(notifier=CapturingNotifier(), inner_sink=base_sink)
    otel_ingester = OTelIngester(sink=otel_sink)

    # feed synthetic error scenario 
    traces_req = create_test_traces()
    metrics_req = create_test_metrics()
    logs_req = create_test_logs()

    otel_ingester.ingest_traces(traces_req)
    otel_ingester.ingest_metrics(metrics_req)
    otel_ingester.ingest_logs(logs_req)

    check("At least one Slack alert was captured from synthetic data",
          len(posted["alerts"]) >= 1,
          f"Captured alerts: {posted['alerts']}")
    print()

    from graph.graph_builder import GraphBuilder
    from graph.context_retriever import ContextRetriever
    from graph.agent import RCAAgent
    from llm_integration.client import MockClient

    gb2 = GraphBuilder()
    gb2._ensure_node("cart-service")
    gb2._ensure_node("frontend")
    gb2.graph.add_edge("frontend", "cart-service", latency=5000)
    gb2.graph.nodes["cart-service"]["status"] = "error"
    gb2.graph.nodes["cart-service"]["recent_events"].append({
        "type": "error_log", "severity": "ERROR",
        "message": "DB timeout", "timestamp": time.time()
    })

    context = ContextRetriever(gb2).get_context("frontend")
    report = RCAAgent(client=MockClient()).analyze(context)

    notifier = CapturingNotifier()
    notifier.post_rca_report("frontend", report)
    check("RCA report posted to Slack", len(posted["reports"]) >= 1)
    print()


# Test live Slack 
def test_live_slack():
    print("── Test: Live Slack (posting real messages) ─────────────────────")
    from RootScout.slack_connector import SlackConfig, SlackNotifier

    token = os.getenv("SLACK_BOT_TOKEN", "")
    channel = os.getenv("SLACK_ALERT_CHANNEL", "#incidents")
    if not token:
        print(f"[{INFO}] Skipped — set SLACK_BOT_TOKEN to run live test\n")
        return

    cfg = SlackConfig(bot_token=token, alert_channel=channel)
    notifier = SlackNotifier(cfg)

    print(f"  Posting test alert to {channel}...")
    notifier.post_incident_alert(
        service="cart-service-TEST",
        status="error",
        signal="trace",
        detail="[RootScout test] Database connection timeout after 5000ms"
    )
    check("Live alert posted without exception", True)

    print(f"  Posting test RCA report to {channel}...")
    notifier.post_rca_report(
        focus_service="frontend-TEST",
        report={
            "root_cause_service": "cart-service-TEST",
            "confidence": 0.95,
            "reasoning": (
                "[RootScout test] The frontend alert is downstream of cart-service. "
                "Traces show 5000ms timeout immediately following deployment a1b2c3d."
            ),
            "recommended_action": "git revert a1b2c3d && kubectl rollout restart deployment/cart-service",
        }
    )
    check("Live RCA report posted without exception", True)
    print()


# Endpoint smoke test 
def test_endpoint():
    print("── Test: /slack/commands endpoint ──────────────────────────────")
    import hashlib, hmac as hmac_mod, time as time_mod
    try:
        import httpx
    except ImportError:
        print(f"[{INFO}] httpx not available, skipping\n")
        return

    base_url = os.getenv("ROOTSCOUT_URL", "http://localhost:8000")
    secret = os.getenv("SLACK_SIGNING_SECRET", "")

    body = b"command=%2Frca&text=cart-service&response_url=https%3A%2F%2Fexample.com%2F"
    timestamp = str(int(time_mod.time()))
    sig_base = f"v0:{timestamp}:{body.decode()}"
    if secret:
        sig = "v0=" + hmac_mod.new(secret.encode(), sig_base.encode(), hashlib.sha256).hexdigest()
    else:
        sig = "v0=dummy"

    with httpx.Client(timeout=5) as client:
        try:
            r = client.post(
                f"{base_url}/slack/commands",
                content=body,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "X-Slack-Request-Timestamp": timestamp,
                    "X-Slack-Signature": sig,
                },
            )
            check(f"/slack/commands returned 200", r.status_code == 200,
                  f"status={r.status_code} body={r.text[:200]}")
            data = r.json()
            check("Response contains 'text' key", "text" in data)
        except httpx.ConnectError:
            print(f"  Could not connect to {base_url}. Is the server running?")
            print(f"  Start it with: python -m uvicorn RootScout.main:create_app --factory --port 8000")
            sys.exit(1)
    print()


# Entry point
if __name__ == "__main__":
    live = "--live" in sys.argv
    endpoint = "--endpoint" in sys.argv

    print("=" * 60)
    print("  RootScout Slack Connector Test Suite")
    print("=" * 60)

    test_config()
    test_block_generation()
    test_alert_sink()
    test_signature_verification()
    test_pipeline_integration()

    if live:
        test_live_slack()
    else:
        print("── Live Slack test (skipped) ─────────────────────────────────")
        print("   Run with --live and SLACK_BOT_TOKEN set to post real messages\n")

    if endpoint:
        test_endpoint()
    else:
        print("── Endpoint test (skipped) ───────────────────────────────────")
        print("   Run with --endpoint (server must be running on :8000)\n")

    print("=" * 60)
    print("  All tests passed.")
    print("=" * 60)
