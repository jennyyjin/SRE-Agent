import json
from typing import Any, Dict, List
from llm_integration.client import MockClient
from graph.data_parser import enrich_context_from_github_output_path


class RCAAgent:
    def __init__(self, client=None, github_output_path=None):
        """
        Initializes the RootScout RCA Agent.

        Args:
            client: LLM client (defaults to MockClient for safety)
            github_output_path: Path to GitHub JSONL file for context enrichment.
                               If not provided, will use GITHUB_OUTPUT_PATH env var.
        """
        self.client = client or MockClient()
        self.github_output_path = github_output_path

    def analyze(self, context_packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a professional Root Cause Analysis (RCA) report.
        Automatically enriches context with GitHub PR/commit data if available.
        """
        # Enrich context using GitHub JSONL (from instance path or GITHUB_OUTPUT_PATH env var)
        context_packet = enrich_context_from_github_output_path(
            context_packet,
            github_output_path=self.github_output_path,
            env_var="GITHUB_OUTPUT_PATH",
            max_events_per_service=25,
            lookback_hours=168,
            verbose=True,
        )

        prompt = self._construct_prompt(context_packet)

        print("[Agent] Prompt constructed. Sending to LLM...")
        response_str = self.client.generate_content(prompt)

        try:
            cleaned = response_str.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            return parsed
        except Exception as e:
            print(f"[Agent] JSON parse failed: {e}")
            print(f"[Agent] Raw response tail: ...{response_str[-200:]}")
            return {"raw_response": response_str, "error": f"Failed to parse JSON: {str(e)}"}

    def _construct_prompt(self, context: Dict[str, Any]) -> str:
        """
        Source-agnostic prompt builder.

        Expects node["events"] to contain envelope events:
          {source, kind, timestamp, summary, payload}
        """
        service_lines: List[str] = []

        # Safeguards for prompt size
        max_events_per_node = 12
        max_patch_chars = 1200

        for node in context.get("related_nodes", []):
            status_str = "ERROR" if node.get("status") == "error" else "OK"
            line = f"- Service: {node.get('service')} [{status_str}]"

            events = node.get("events") or []
            for e in events[:max_events_per_node]:
                src = e.get("source", "unknown")
                kind = e.get("kind", "event")
                ts = e.get("timestamp")
                summary = e.get("summary") or ""

                line += f"\n  - [{src}/{kind}] {summary}".rstrip()
                if ts:
                    line += f" at {ts}"

                payload = e.get("payload") or {}
                if isinstance(payload, dict):
                    # Helpful fields if present (GitHub, but harmless for others)
                    if payload.get("filename"):
                        line += f"\n    filename: {payload.get('filename')}"
                    if payload.get("status") is not None:
                        adds = int(payload.get("additions") or 0)
                        dels = int(payload.get("deletions") or 0)
                        line += f"\n    status: {payload.get('status')} (+{adds}/-{dels})"
                    if payload.get("sha"):
                        line += f"\n    sha: {payload.get('sha')}"

                    patch = payload.get("patch")
                    if patch:
                        snippet = patch[:max_patch_chars]
                        line += f"\n    patch:\n{snippet}"
                        if len(patch) > max_patch_chars:
                            line += "\n    [patch truncated]"

            service_lines.append(line)

        context_str = "\n".join(service_lines)

        return f"""
### SYSTEM ROLE
You are the Lead On-Call Site Reliability Engineer (SRE) for RootScout.
Your goal is to investigate outages in distributed systems and identify "Patient Zero."
You are analytical, data-driven, and focused on minimizing Mean Time to Recovery (MTTR).

### INCIDENT CONTEXT
An alert has fired on the focus service: **{context.get('focus_service')}**.
The following dependency graph and recent events have been retrieved.
All timestamps are in UTC+8 (CST).

{context_str}

### INVESTIGATION TASK
Analyze the topology and event data to:
1. Identify the root cause service (where the failure originated).
2. Estimate when the fault first began, based on the earliest anomalous events you can observe.
3. Determine if a specific change (deployment, PR, config, etc.) is the likely trigger.
4. Provide a clear reasoning for how the failure propagated.
5. Suggest a specific remediation command (e.g., git revert, kubectl rollout undo, disable feature flag).

### RESPONSE FORMAT
Return ONLY a valid JSON object with the following structure:
{{
  "root_cause_service": "<service_name>",
  "root_cause_datetime": "<YYYY-MM-DD HH:MM:SS — earliest timestamp when the fault began, in UTC+8>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<professional SRE explanation>",
  "recommended_action": "<specific command to fix the issue>"
}}
""".strip()
