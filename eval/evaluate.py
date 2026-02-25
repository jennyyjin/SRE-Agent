"""
evaluate.py - OpenRCA-compatible evaluator with semantic similarity for reason scoring.

Adapted from: https://github.com/microsoft/OpenRCA/blob/main/main/evaluate.py
Extensions:
  - Semantic similarity scoring for 'root cause reason' (cosine sim ≥ threshold)
    in addition to exact match, so LLM paraphrases still score correctly.
  - Falls back to exact match if sentence-transformers is unavailable.
"""

import re
import os
import itertools
from datetime import datetime


# ---------------------------------------------------------------------------
# Semantic similarity (optional)
# ---------------------------------------------------------------------------

_encoder = None
_SIM_THRESHOLD = 0.50  # cosine similarity threshold for reason match

def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        _encoder = None
    return _encoder


def semantic_sim_score(pred: str, ref: str) -> float:
    """
    Return the raw cosine similarity score between pred and ref (0.0–1.0).

    Uses the same sentence-splitting strategy as _semantic_match: the score
    is the maximum similarity across all sentences in pred vs the ref phrase.
    Returns 1.0 for exact/substring matches, 0.0 if either string is empty
    or sentence-transformers is unavailable and there is no substring match.
    """
    if not pred or not ref:
        return 0.0

    pred_clean = pred.strip()
    ref_clean = ref.strip().lower()

    if pred_clean.lower() == ref_clean:
        return 1.0
    if ref_clean in pred_clean.lower():
        return 1.0

    encoder = _get_encoder()
    if encoder is None:
        return 1.0 if ref_clean in pred_clean.lower() else 0.0

    import numpy as np
    import re as _re
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', pred_clean) if s.strip()]
    if not sentences:
        sentences = [pred_clean]

    ref_emb = encoder.encode([ref_clean], normalize_embeddings=True)[0]
    sent_embs = encoder.encode(sentences, normalize_embeddings=True)
    return float(max(np.dot(sent_embs, ref_emb)))


def _semantic_match(pred: str, ref: str) -> bool:
    """Return True if pred and ref are semantically similar (score ≥ threshold)."""
    return semantic_sim_score(pred, ref) >= _SIM_THRESHOLD


# ---------------------------------------------------------------------------
# Core evaluate() — mirrors OpenRCA's signature exactly
# ---------------------------------------------------------------------------

def evaluate(prediction: str, scoring_points: str):
    """
    Evaluate a single JSON-like prediction against OpenRCA scoring_points.

    Returns: (passing_criteria, failing_criteria, score, details)
      - passing_criteria: list of matched criteria strings
      - failing_criteria: list of unmatched criteria strings
      - score: float in [0, 1]
      - details: list of per-criterion dicts, each with keys:
            type        "component" | "reason" | "time"
            predicted   the string the model gave
            expected    the reference string from scoring_points
            similarity  float cosine sim for reason criteria, None otherwise
            passed      bool
    """

    predict_pattern = (
        r'{\s*'
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    predict_matches = re.findall(predict_pattern, prediction)

    predict_results = []
    for match in predict_matches:
        datetime_str, component, reason = match
        predict_results.append({
            "root cause occurrence datetime": datetime_str,
            "root cause component": component,
            "root cause reason": reason,
        })

    prediction_length = len(predict_results)

    component_pattern = r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"
    reason_pattern = r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"
    time_pattern = (
        r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes "
        r"\(i\.e\., <=1min\) of ([^\n]+)"
    )

    components = re.findall(component_pattern, scoring_points)
    reasons = re.findall(reason_pattern, scoring_points)
    times = re.findall(time_pattern, scoring_points)

    scoringpoints_length = max(len(components), len(reasons), len(times), 1)
    scores_num = len(components) + len(reasons) + len(times)

    if scores_num == 0:
        return [], [], 0.0, []

    def _time_within_60s(ref_str: str, pred_str: str) -> bool:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            t1 = datetime.strptime(ref_str.strip(), fmt)
            t2 = datetime.strptime(pred_str.strip(), fmt)
            return abs((t1 - t2).total_seconds()) <= 60
        except ValueError:
            return False

    scores_get = 0
    passing_criteria = []
    failing_criteria = []
    best_details: list = []

    if scoringpoints_length == prediction_length:
        best_score = -1
        for perm in itertools.permutations(predict_results):
            current_score = 0
            current_passing = []
            current_details = []
            for i in range(scoringpoints_length):
                if len(components) == scoringpoints_length:
                    matched = perm[i]["root cause component"].strip() == components[i].strip()
                    current_details.append({
                        "type": "component",
                        "predicted": perm[i]["root cause component"],
                        "expected": components[i],
                        "similarity": None,
                        "passed": matched,
                    })
                    if matched:
                        current_score += 1
                        current_passing.append(f"component:{components[i]}")

                if len(reasons) == scoringpoints_length:
                    sim = semantic_sim_score(perm[i]["root cause reason"], reasons[i])
                    matched = sim >= _SIM_THRESHOLD
                    current_details.append({
                        "type": "reason",
                        "predicted": perm[i]["root cause reason"],
                        "expected": reasons[i],
                        "similarity": sim,
                        "passed": matched,
                    })
                    if matched:
                        current_score += 1
                        current_passing.append(f"reason:{reasons[i]}")

                if len(times) == scoringpoints_length:
                    matched = _time_within_60s(times[i], perm[i]["root cause occurrence datetime"])
                    current_details.append({
                        "type": "time",
                        "predicted": perm[i]["root cause occurrence datetime"],
                        "expected": times[i],
                        "similarity": None,
                        "passed": matched,
                    })
                    if matched:
                        current_score += 1
                        current_passing.append(f"time:{times[i]}")

            if current_score > best_score:
                best_score = current_score
                passing_criteria = current_passing
                best_details = current_details

        scores_get = best_score

    all_criteria = (
        [f"component:{c}" for c in components]
        + [f"reason:{r}" for r in reasons]
        + [f"time:{t}" for t in times]
    )
    failing_criteria = list(set(all_criteria) - set(passing_criteria))
    final_score = scores_get / scores_num
    return passing_criteria, failing_criteria, round(final_score, 4), best_details


# ---------------------------------------------------------------------------
# File-level evaluate — same interface as OpenRCA
# ---------------------------------------------------------------------------

def file_evaluate(prediction_file: str, query_file: str, report_file: str):
    import pandas as pd

    pred_df = pd.read_csv(prediction_file)
    query_df = pd.read_csv(query_file)

    if len(pred_df) != len(query_df):
        raise ValueError("Prediction file and query file must have the same number of rows.")

    rows = []
    for idx in range(len(pred_df)):
        prediction = pred_df.loc[idx, "prediction"]
        scoring_pts = query_df.loc[idx, "scoring_points"]
        instruction = query_df.loc[idx, "instruction"]
        task_index = query_df.loc[idx, "task_index"]

        passing, failing, score, _details = evaluate(str(prediction), str(scoring_pts))
        rows.append({
            "query": instruction,
            "answer": prediction,
            "groundtruth": scoring_pts,
            "passed": passing,
            "failed": failing,
            "score": score,
            "task_index": task_index,
        })

    eval_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    if os.path.exists(report_file):
        eval_df.to_csv(report_file, mode="a", header=False, index=False)
    else:
        eval_df.to_csv(report_file, index=False)


# ---------------------------------------------------------------------------
# Report printer — mirrors OpenRCA's report()
# ---------------------------------------------------------------------------

def report(report_file: str):
    import pandas as pd

    df = pd.read_csv(report_file)

    def _difficulty(task_index: str) -> str:
        try:
            n = int(task_index.split("_")[1])
        except (IndexError, ValueError):
            return "unknown"
        if n <= 3:
            return "easy"
        if n <= 6:
            return "medium"
        return "hard"

    df["difficulty"] = df["task_index"].apply(_difficulty)

    w = 14
    print(f"\n{'─'*56}")
    print(f"{'Class':<{w}}{'Total(#)':<{w}}{'Correct(#)':<{w}}{'Accuracy(%)':<{w}}")
    print(f"{'─'*56}")

    overall_total = 0
    overall_correct = 0

    for tier in ["easy", "medium", "hard"]:
        subset = df[df["difficulty"] == tier]
        total = len(subset)
        correct = len(subset[subset["score"] == 1.0])
        overall_total += total
        overall_correct += correct
        acc = f"{correct/total:.1%}" if total > 0 else "N/A"
        print(f"{tier:<{w}}{total:<{w}}{correct:<{w}}{acc:<{w}}")

    print(f"{'─'*56}")
    acc_all = f"{overall_correct/overall_total:.1%}" if overall_total > 0 else "N/A"
    print(f"{'Total':<{w}}{overall_total:<{w}}{overall_correct:<{w}}{acc_all:<{w}}")
    print(f"{'─'*56}\n")

    # Per-scenario detail
    print("Per-scenario scores:")
    for _, row in df.iterrows():
        mark = "PASS" if row["score"] == 1.0 else f"{row['score']:.2f}"
        print(f"  [{row['task_index']}] {row['query'][:60]:<62} {mark}")


# ---------------------------------------------------------------------------
# CLI entry point (mirrors OpenRCA usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, nargs="+", help="prediction CSV files")
    parser.add_argument("-q", type=str, nargs="+", help="query/groundtruth CSV files")
    parser.add_argument("-r", type=str, help="report output CSV")
    args = parser.parse_args()

    if len(args.p) != len(args.q):
        raise ValueError("Must provide equal numbers of -p and -q files.")

    if os.path.exists(args.r):
        os.remove(args.r)

    for p_file, q_file in zip(args.p, args.q):
        try:
            file_evaluate(p_file, q_file, args.r)
        except Exception as e:
            print(f"Error evaluating {p_file}: {e}")

    report(args.r)
