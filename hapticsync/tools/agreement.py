#!/usr/bin/env python3
"""
agreement.py — Evaluate pipeline agreement vs. offline reference decode (Section 12.1)
Metrics:
- Precision & Recall of phoneme onsets within +/- 40 ms window
- Phoneme sequence Levenshtein edit distance & PER
"""

import sys
import json


def levenshtein_distance(seq1, seq2):
    """Compute Levenshtein distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[n][m]


def evaluate_agreement(ref_events, test_events, tolerance_s=0.040):
    """
    Evaluate precision and recall within +/- tolerance_s (default 40 ms).
    """
    matched_ref = set()
    matched_test = set()

    # Match based on timestamp and label
    for t_idx, t_evt in enumerate(test_events):
        t_time = t_evt.get("t_center_s", (t_evt.get("abs_frame", 0) * 320 + 200) / 16000.0)
        t_label = t_evt.get("phoneme")

        best_ref_idx = None
        best_dt = tolerance_s

        for r_idx, r_evt in enumerate(ref_events):
            if r_idx in matched_ref:
                continue
            r_time = r_evt.get("t_center_s", (r_evt.get("abs_frame", 0) * 320 + 200) / 16000.0)
            r_label = r_evt.get("phoneme")

            if t_label == r_label:
                dt = abs(t_time - r_time)
                if dt <= best_dt:
                    best_dt = dt
                    best_ref_idx = r_idx

        if best_ref_idx is not None:
            matched_ref.add(best_ref_idx)
            matched_test.add(t_idx)

    tp = len(matched_test)
    fp = len(test_events) - tp
    fn = len(ref_events) - len(matched_ref)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # Sequence level edit distance
    seq_ref = [e.get("phoneme") for e in ref_events]
    seq_test = [e.get("phoneme") for e in test_events]
    edit_dist = levenshtein_distance(seq_ref, seq_test)
    per = edit_dist / max(len(seq_ref), 1)

    return {
        "ref_count": len(ref_events),
        "test_count": len(test_events),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "edit_distance": edit_dist,
        "per": round(per, 4),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 agreement.py <reference_events.json> <pipeline_events.json>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        ref_data = json.load(f)
    with open(sys.argv[2], "r") as f:
        test_data = json.load(f)

    ref_events = ref_data.get("events", ref_data)
    test_events = test_data.get("events", test_data)

    results = evaluate_agreement(ref_events, test_events)

    print("── Phoneme Agreement Evaluation Report ────────────────────")
    print(f"Reference events: {results['ref_count']}")
    print(f"Pipeline events : {results['test_count']}")
    print(f"Matched onsets  : {results['true_positives']}")
    print(f"Precision       : {results['precision'] * 100:.2f}%")
    print(f"Recall          : {results['recall'] * 100:.2f}%")
    print(f"F1 Score        : {results['f1_score']:.4f}")
    print(f"Edit Distance   : {results['edit_distance']}")
    print(f"Phoneme Err Rate: {results['per'] * 100:.2f}%")
    print("───────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()

