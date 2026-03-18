# RootScout

RootScout is an agentic system for automated root cause analysis (RCA) in distributed systems. It ingests telemetry (OTel traces, metrics, logs) and GitHub PR data, builds a causal dependency graph, and uses an LLM to identify which service caused an incident and why.

## How it works

1. **Graph construction** — Trace spans are ingested and wired into a directed dependency graph. Each node tracks health status and recent events.
2. **Fault isolation** — When an alert fires, BFS traversal from the alerting service collects the subgraph of suspects.
3. **LLM reasoning** — A Gemini (or Claude) agent receives the context packet and returns a structured root cause report.

## Setup

### Prerequisites

- Python 3.9+
- Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Install

```bash
git clone https://github.com/asthamohta/CS224G-SRE.git
cd CS224G-SRE
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### Run the demo

```bash
python demo.py
```

The demo ingests synthetic OTel data, builds a dependency graph, and runs LLM-powered RCA on a simulated cart-service failure.

---

## Evaluation

Two evaluation tracks measure whether the agent correctly identifies the component, reason, and datetime of a fault, using [OpenRCA](https://github.com/microsoft/OpenRCA) scoring.

Install eval dependencies:

```bash
pip install -r requirements_eval.txt
```

### Scoring

Each incident is scored on up to three criteria depending on the task type:

| Criterion | Match method |
|---|---|
| Root cause component | Exact string match |
| Root cause reason | Cosine similarity >= 0.50 (all-MiniLM-L6-v2) |
| Occurrence datetime | Within +/- 60 s of ground truth |

A scenario passes only when every applicable criterion is met.

---

### Track A — Synthetic benchmark

Ten hand-crafted scenarios with known topology and injected faults. Useful for iterating on the agent prompt without running against real data.

```bash
python eval/run_eval.py              # all 10 scenarios (requires Gemini API key)
python eval/run_eval.py --mock       # mock LLM, no API key needed
python eval/run_eval.py --difficulty easy
```

Sample result:

```
Class         Total     Correct   Accuracy
easy          3         2         66.7%
medium        3         3         100.0%
hard          4         3         75.0%
Total         10        8         80.0%
```

---

### Track B — Real OpenRCA Bank telemetry

27 incidents from the [OpenRCA Bank dataset](https://github.com/microsoft/OpenRCA) — a Java-based banking microservices system with 14 pods. Requires the `Bank/` dataset directory at the project root:

```
Bank/
  query.csv
  record.csv
  telemetry/
    2021_03_04/
      metric/metric_container.csv
      log/log_service.csv
    2021_03_06/ ...
```

```bash
python eval/run_openrca_eval.py              # 27 Bank incidents (requires Gemini API key)
python eval/run_openrca_eval.py --mock       # no API key needed
python eval/run_openrca_eval.py --n 5        # quick test with 5 incidents
python eval/run_openrca_eval.py --bank-dir /path/to/Bank
```

Sample result:

```
BANK BENCHMARK SUMMARY  (real OpenRCA telemetry)
Class         Total     Full pass   Avg score
easy          2         1           0.71
medium        18        4           0.52
hard          7         1           0.38
Total         27        6           0.49
```

Scores are lower on real data than synthetic because real incidents have noisy signals, cross-pod resource contention, and ambiguous telemetry.

---

## Known limitations

- **Datetime scoring on Track B is not genuine.** The fault timestamp is taken directly from `record.csv` rather than predicted by the agent, so datetime criteria always pass. A real fix would add a `root_cause_datetime` field to the agent's response schema.
- **No trace topology on real data.** `trace_span.csv` uses internal container IDs that don't map to pod names, so a static hand-written topology is used instead.
- **Noisy anomaly detection.** KPI thresholds are heuristic; during real incidents many pods spike simultaneously, making causal isolation harder.
- **Single system.** Only the Bank system is evaluated. OpenRCA also includes Telecom and Market.

---

## Slack Connector

RootScout posts incident alerts and RCA reports to Slack via a Slack Bot. The connector lives in `RootScout/slack_connector.py`.

### Setup

**1. Create a Slack Bot**

- Go to [api.slack.com/apps](https://api.slack.com/apps) -> **Create New App -> From scratch**
- Under **OAuth & Permissions**, add bot scopes: `chat:write`, `chat:write.public`
- **Install to Workspace** and copy the **Bot OAuth Token** (`xoxb-...`)

**2. Add to `.env`**

```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token    
SLACK_ALERT_CHANNEL=#incidents           # default: #incidents
SLACK_RCA_CHANNEL=                       # defaults to alert channel
SLACK_SIGNING_SECRET=                    # for slash command verification
SLACK_ALERT_COOLDOWN_SECONDS=300         # seconds between repeat alerts per service
```

**3. Run the demo**

```bash
python demo_slack.py
```

Without `SLACK_BOT_TOKEN` it runs in dry-run mode (prints what would be sent). With the token it posts real messages 

### Usage

```python
from RootScout.slack_connector import SlackConfig, SlackNotifier

notifier = SlackNotifier(SlackConfig(bot_token="xoxb-..."))

# Incident alert
notifier.post_incident_alert(service="cart-service", status="error", signal="trace")

# RCA report
notifier.post_rca_report(focus_service="cart-service", report={
    "root_cause_service": "cart-service",
    "confidence": 0.92,
    "reasoning": "...",
    "recommended_action": "kubectl rollout undo deployment/cart-service",
})
```

Wrap any `TelemetrySink` with `SlackAlertSink` to auto-alert on ERROR telemetry:

```python
from RootScout.slack_connector import SlackAlertSink

sink = SlackAlertSink(notifier=notifier, inner_sink=your_existing_sink)
```

### Tests

```bash
python test_slack_connector.py             # unit + integration (no network)
python test_slack_connector.py --live      # posts real messages (needs SLACK_BOT_TOKEN)
python test_slack_connector.py --endpoint  # smoke-tests /slack/commands (server must be running)
```

---

## Project layout

```
graph/             Graph construction, context retrieval, RCA agent
llm_integration/   Gemini and Claude client wrappers
eval/              Evaluation scripts and scenarios
RootScout/         OTel ingester service (FastAPI)
Ingester/          GitHub webhook ingester
slack_integration/ Legacy Slack webhook notifier
```
