"""
Slack integration for RootScout. 

- Sends incident alerts to Slack when services enter error states
- Posts structured RCA summaries and reports to Slack
- Handles /rca slash commands to trigger on-demand analysis
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

import httpx

from RootScout.otel_ingester import TelemetrySink


@dataclass
class SlackConfig:
    bot_token: str
    signing_secret: str = ""
    alert_channel: str = "#incidents"
    # where RCA reports are posted, falls back to alert_channel if empty
    rca_channel: str = ""
    # minimum seconds between repeated alerts for the same service
    alert_cooldown_seconds: int = 300


def slack_config_from_env() -> Optional[SlackConfig]:
    """
    Reads Slack config from environment variables.
    Returns None if SLACK_BOT_TOKEN is not set (Slack integration disabled).
    """
    token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if not token:
        return None
    return SlackConfig(
        bot_token=token,
        signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
        alert_channel=os.getenv("SLACK_ALERT_CHANNEL", "#incidents"),
        rca_channel=os.getenv("SLACK_RCA_CHANNEL", ""),
        alert_cooldown_seconds=int(os.getenv("SLACK_ALERT_COOLDOWN_SECONDS", "300")),
    )


class SlackClient:
    """
    Wrapper around Slack Web API using httpx.
    Uses the bot token for authentication. 
    """

    _BASE = "https://slack.com/api"

    def __init__(self, bot_token: str) -> None:
        self._token = bot_token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def post_message(
        self, channel: str, text: str, blocks: Optional[list] = None
    ) -> Dict[str, Any]:
        """Synchronous chat.postMessage call."""
        payload: Dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            payload["blocks"] = blocks
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{self._BASE}/chat.postMessage",
                headers=self._headers(),
                content=json.dumps(payload),
            )
            r.raise_for_status()
            return r.json()

    async def async_post_message(
        self, channel: str, text: str, blocks: Optional[list] = None
    ) -> Dict[str, Any]:
        """Async chat.postMessage call."""
        payload: Dict[str, Any] = {"channel": channel, "text": text}
        if blocks:
            payload["blocks"] = blocks
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{self._BASE}/chat.postMessage",
                headers=self._headers(),
                content=json.dumps(payload),
            )
            r.raise_for_status()
            return r.json()


class SlackNotifier:
    """
    Formats and posts incident alerts and RCA reports to Slack using Block Kit.
    """

    def __init__(self, config: SlackConfig) -> None:
        self._config = config
        self._client = SlackClient(config.bot_token)

    def post_incident_alert(
        self, service: str, status: str, signal: str, detail: str = ""
    ) -> None:
        """Posts an incident alert to the configured alert channel."""
        channel = self._config.alert_channel
        text = f"Incident: {service} is {status.upper()}"
        blocks = self._build_alert_blocks(service, status, signal, detail)
        self._safe_post(channel, text, blocks, label="alert")

    def post_rca_report(self, focus_service: str, report: Dict[str, Any]) -> None:
        """Posts a formatted RCA report to the configured RCA channel."""
        channel = self._config.rca_channel or self._config.alert_channel
        text = f"RCA Report: {focus_service}"
        blocks = self._build_rca_blocks(focus_service, report)
        self._safe_post(channel, text, blocks, label="rca-report")

    def _build_alert_blocks(
        self, service: str, status: str, signal: str, detail: str
    ) -> list:
        emoji = ""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        blocks: list = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Incident Alert: {service}",
                    "emoji": False,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Service:*\n`{service}`"},
                    {"type": "mrkdwn", "text": f"*Status:*\n{status.upper()}"},
                    {"type": "mrkdwn", "text": f"*Signal:*\n{signal}"},
                    {"type": "mrkdwn", "text": f"*Detected at:*\n{ts}"},
                ],
            },
        ]

        if detail:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Detail:*\n{detail[:300]}"},
                }
            )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"RootScout detected an error signal. "
                            f"Trigger full RCA with `/rca {service}`"
                        ),
                    }
                ],
            }
        )
        return blocks

    def _build_rca_blocks(
        self, focus_service: str, report: Dict[str, Any]
    ) -> list:
        root_cause = report.get("root_cause_service", "unknown")
        confidence = float(report.get("confidence", 0.0))
        reasoning = report.get("reasoning", "No reasoning provided.")
        action = report.get("recommended_action", "")

        confidence_pct = f"{confidence * 100:.0f}%"
        conf_emoji = ""

        blocks: list = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"RCA Report: {focus_service}",
                    "emoji": False,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Root Cause Service:*\n`{root_cause}`"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n{confidence_pct}",
                    },
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    # block Kit section text cap: 3000 chars
                    "text": f"*Reasoning:*\n{reasoning[:2900]}",
                },
            },
        ]

        if action:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Recommended Action:*\n```{action}```",
                    },
                }
            )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "Generated by *RootScout* RCA Agent"}
                ],
            }
        )
        return blocks

    def _safe_post(
        self, channel: str, text: str, blocks: list, label: str
    ) -> None:
        try:
            resp = self._client.post_message(channel=channel, text=text, blocks=blocks)
            if not resp.get("ok"):
                print(f"[SlackNotifier] {label} post error: {resp.get('error')}")
        except Exception as exc:
            print(f"[SlackNotifier] Failed to post {label}: {exc}")


class SlackAlertSink(TelemetrySink):
    """
    TelemetrySink decorator that posts a Slack alert whenever a telemetry
    record carries an ERROR status, then forwards the record to the inner sink.

    Per-service cooldown prevents duplicate alerts for the same ongoing incident.
    """

    def __init__(
        self,
        notifier: SlackNotifier,
        inner_sink: Optional[TelemetrySink] = None,
    ) -> None:
        self._notifier = notifier
        self._inner = inner_sink
        # service_name -> monotonic timestamp of last alert
        self._last_alert: Dict[str, float] = {}

    def emit(self, record: Dict[str, Any]) -> None:
        # forward to the inner sink first so nothing is lost
        if self._inner:
            self._inner.emit(record)

        # OTLP status_code 2 == ERROR
        is_error = (
            record.get("status_code") == 2
            or str(record.get("status", "")).lower() == "error"
        )
        if not is_error:
            return

        service = (
            record.get("service")
            or record.get("service_name")
            or "unknown"
        )
        signal = record.get("signal", "trace")

        # cooldown check
        now = time.monotonic()
        cooldown = self._notifier._config.alert_cooldown_seconds
        if now - self._last_alert.get(service, 0.0) < cooldown:
            return

        self._last_alert[service] = now

        # build a human-readable detail line from the record
        detail = (
            record.get("status_message")
            or (f"span: {record['name']}" if record.get("name") else "")
        )

        self._notifier.post_incident_alert(
            service=service,
            status="error",
            signal=signal,
            detail=detail,
        )


class SlackCommandHandler:
    """
    Validates incoming Slack slash-command requests and dispatches the RCA
    agent in the background, posting the report back to Slack when done.

    Supported commands: /rca <service_name> - runs RCA for the named service
    """

    def __init__(
        self,
        config: SlackConfig,
        graph_builder=None,
        rca_agent=None,
    ) -> None:
        self._config = config
        self._notifier = SlackNotifier(config)
        self._graph_builder = graph_builder
        self._rca_agent = rca_agent

    # Signature verification
    def verify_signature(
        self, raw_body: bytes, timestamp: str, signature: str
    ) -> bool:
        """
        Verifies a Slack request using the signing secret.
        """
        if not self._config.signing_secret:
            return True  

        try:
            if abs(time.time() - float(timestamp)) > 300:
                return False
        except (ValueError, TypeError):
            return False

        sig_base = f"v0:{timestamp}:{raw_body.decode('utf-8')}"
        expected = "v0=" + hmac.new(
            self._config.signing_secret.encode("utf-8"),
            sig_base.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)


    async def handle(self, request: Any, background_tasks: Any) -> Dict[str, Any]:
        """
        Entry point for POST /slack/commands.

        Returns an immediate acknowledgement and 
        dispatches any heavy work to background_tasks.
        """
        from fastapi import HTTPException

        raw = await request.body()
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")

        if not self.verify_signature(raw, timestamp, signature):
            raise HTTPException(status_code=401, detail="Invalid Slack request signature")

        # Slack sends application/x-www-form-urlencoded
        params = {k: v[0] for k, v in parse_qs(raw.decode("utf-8")).items()}
        command = params.get("command", "")
        text = params.get("text", "").strip()
        response_url = params.get("response_url", "")

        if command == "/rca":
            service_name = text or "unknown"
            background_tasks.add_task(
                self._run_rca_and_post, service_name, response_url
            )
            return {
                "response_type": "in_channel",
                "text": (
                    f"Running RCA for `{service_name}`... "
                    "results will be posted to Slack shortly."
                ),
            }

        return {"text": f"Unknown command: `{command}`. Supported: `/rca <service>`"}


    async def _run_rca_and_post(
        self, service_name: str, response_url: str
    ) -> None:
        """
        Runs RCA for service_name, posts the report to Slack, and 
        follows up on the slash-command response_url with a summary.
        """
        try:
            if not self._graph_builder:
                self._notifier.post_rca_report(
                    service_name,
                    {
                        "root_cause_service": "unknown",
                        "confidence": 0.0,
                        "reasoning": (
                            "Graph builder not enabled. "
                            "Set ENABLE_GRAPH_BUILDER=true and restart the server."
                        ),
                        "recommended_action": "",
                    },
                )
                return

            from graph.context_retriever import ContextRetriever
            from graph.agent import RCAAgent

            context_packet = ContextRetriever(self._graph_builder).get_context(
                service_name
            )

            if "error" in context_packet:
                self._notifier.post_rca_report(
                    service_name,
                    {
                        "root_cause_service": "unknown",
                        "confidence": 0.0,
                        "reasoning": context_packet["error"],
                        "recommended_action": "",
                    },
                )
                return

            agent = self._rca_agent or RCAAgent()
            report = agent.analyze(context_packet)
            self._notifier.post_rca_report(service_name, report)

            # acknowledge back to the Slack user via response_url
            if response_url:
                channel = self._config.rca_channel or self._config.alert_channel
                await self._post_response_url(
                    response_url,
                    f":white_check_mark: RCA complete for `{service_name}`. "
                    f"Full report posted to {channel}.",
                )

        except Exception as exc:
            print(f"[SlackCommandHandler] RCA failed for {service_name!r}: {exc}")
            if response_url:
                await self._post_response_url(
                    response_url, f":x: RCA failed: {exc}"
                )

    @staticmethod
    async def _post_response_url(url: str, text: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json={"text": text})
        except Exception as exc:
            print(f"[SlackCommandHandler] Failed to post to response_url: {exc}")
