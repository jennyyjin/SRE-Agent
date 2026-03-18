# data_parser.py
#
# Parses GitHub JSONL from GITHUB_OUTPUT_PATH and enriches a context packet with
# flexible, source-agnostic "event envelopes" that can support GitHub and future ingestors.
#
# Envelope format:
# {
#   "source": "<string>",
#   "kind": "<string>",
#   "timestamp": "<iso string | None>",
#   "summary": "<string | None>",
#   "payload": { ... }   # source-specific details (can include patch)
# }

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def safe_load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file where each line is a JSON object.
    Skips malformed lines.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def make_envelope(
    *,
    source: str,
    kind: str,
    timestamp: Optional[str],
    summary: Optional[str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "source": source,
        "kind": kind,
        "timestamp": timestamp,
        "summary": summary,
        "payload": payload,
    }


def github_changeevent_to_file_envelopes(change_event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert one GitHub ChangeEvent JSON object (from github_ingester sink) into 1+ envelope events.
    Emits one envelope per changed file, preserving 'patch' in payload.
    """
    ts = change_event.get("ingested_at")
    event_type = change_event.get("event_type") or "github_event"

    repo_owner = change_event.get("repo_owner")
    repo_name = change_event.get("repo_name")
    repo = f"{repo_owner}/{repo_name}" if repo_owner and repo_name else None

    commit_sha = change_event.get("commit_sha")
    pr_number = change_event.get("pr_number")
    title = change_event.get("title")
    url = change_event.get("url")
    service_id = change_event.get("service_id")
    watch_path_prefix = change_event.get("watch_path_prefix")

    files = change_event.get("files") or []
    if not isinstance(files, list):
        files = []

    envelopes: List[Dict[str, Any]] = []

    # If files are missing, still emit a single meta envelope
    if not files:
        payload = {
            "event_type": event_type,
            "repo": repo,
            "service_id": service_id,
            "watch_path_prefix": watch_path_prefix,
            "commit_sha": commit_sha,
            "pr_number": pr_number,
            "title": title,
            "url": url,
        }
        envelopes.append(
            make_envelope(
                source="github",
                kind="change_meta",
                timestamp=ts,
                summary=f"{event_type} observed (no files list)",
                payload=payload,
            )
        )
        return envelopes

    for f in files:
        if not isinstance(f, dict):
            continue

        filename = f.get("filename") or f.get("path") or "unknown"
        status = f.get("status") or "unknown"
        additions = int(f.get("additions") or 0)
        deletions = int(f.get("deletions") or 0)

        payload = dict(f)  # includes patch
        payload["_meta"] = {
            "event_type": event_type,
            "repo": repo,
            "service_id": service_id,
            "watch_path_prefix": watch_path_prefix,
            "commit_sha": commit_sha,
            "pr_number": pr_number,
            "title": title,
            "url": url,
        }

        envelopes.append(
            make_envelope(
                source="github",
                kind="code_change",
                timestamp=ts,
                summary=f"{status}: {filename} (+{additions}/-{deletions})",
                payload=payload,
            )
        )

    return envelopes


def enrich_context_from_github_output_path(
    context_packet: Dict[str, Any],
    *,
    github_output_path: Optional[str] = None,
    env_var: str = "GITHUB_OUTPUT_PATH",
    max_events_per_service: int = 25,
    lookback_hours: int = 168,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Reads GitHub JSONL and attaches envelope events to context packet nodes.
    Join key: node["service"] == change_event["service_id"].

    Args:
        context_packet: The context packet to enrich
        github_output_path: Direct path to JSONL file (takes precedence over env_var)
        env_var: Environment variable name to read path from (default: GITHUB_OUTPUT_PATH)
        max_events_per_service: Max events to attach per service (default: 25)
        lookback_hours: Only include events from last N hours (default: 168 = 1 week)
        verbose: Print status messages
    """
    # Use direct path if provided, otherwise fall back to env var
    path = github_output_path or os.getenv(env_var, "").strip()
    if not path:
        if verbose:
            print(f"ℹ️ [DataParser] No github_output_path provided and {env_var} not set; skipping GitHub enrichment")
        return context_packet

    if not os.path.exists(path):
        if verbose:
            print(f"[DataParser] {env_var} set but file missing: {path}")
        return context_packet

    try:
        raw_change_events = safe_load_jsonl(path)
    except Exception as e:
        if verbose:
            print(f"[DataParser] Failed reading JSONL at {path}: {e}")
        return context_packet

    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - (lookback_hours * 3600)

    by_service: Dict[str, List[Dict[str, Any]]] = {}

    for ce in raw_change_events:
        service_id = ce.get("service_id")
        if not service_id:
            continue

        dt = _parse_iso(ce.get("ingested_at") or "")
        if dt and dt.timestamp() < cutoff:
            continue

        envs = github_changeevent_to_file_envelopes(ce)
        if envs:
            by_service.setdefault(service_id, []).extend(envs)

    # Sort newest-first per service
    for svc, envs in by_service.items():
        envs.sort(key=lambda e: (_parse_iso(e.get("timestamp") or "") or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        if max_events_per_service > 0:
            by_service[svc] = envs[:max_events_per_service]

    out = dict(context_packet)
    out_nodes: List[Dict[str, Any]] = []

    for node in context_packet.get("related_nodes", []):
        n = dict(node)
        svc = n.get("service")
        existing = list(n.get("events") or [])
        additions = by_service.get(svc, [])
        n["events"] = existing + additions
        out_nodes.append(n)

    out["related_nodes"] = out_nodes

    if verbose:
        print(f"[DataParser] Attached GitHub events from {path}")

    return out
