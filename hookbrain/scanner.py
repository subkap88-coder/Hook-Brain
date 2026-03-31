"""
TRIBE v2 wrapper.
Import for compute_metrics() / compute_viral_score().
Run as a script: python scanner.py "hook text" output.json
"""

import sys
import json
import numpy as np

N_TOP = 100


def compute_metrics(preds):
    """Raw TRIBE predictions (T x 20484) → list of per-second dicts."""
    n_hemi = preds.shape[1] // 2
    seconds = []
    for i in range(preds.shape[0]):
        row = preds[i]
        top_idx = np.argsort(row)[-N_TOP:]
        seconds.append({
            "t":               i,
            "mean":            round(float(row.mean()), 4),
            "max":             round(float(row.max()),  4),
            "min":             round(float(row.min()),  4),
            "left_mean":       round(float(row[:n_hemi].mean()), 4),
            "right_mean":      round(float(row[n_hemi:].mean()), 4),
            "top100_mean":     round(float(np.sort(row)[-N_TOP:].mean()), 4),
            "top100_left_pct": int(sum(1 for idx in top_idx if idx < n_hemi)),
        })
    return seconds


def compute_viral_score(seconds):
    """Viral score breakdown from per-second brain data."""
    def get(t, key):
        return seconds[t].get(key, 0) if t < len(seconds) else 0

    watch_signal    = get(0, "mean") + get(1, "mean")
    emotional_onset = get(0, "top100_mean")
    right_dom_t0    = 100 - get(0, "top100_left_pct")
    dropoff_risk    = get(3, "top100_left_pct")
    mean_sustained  = sum(s["mean"] for s in seconds) / len(seconds) if seconds else 0

    viral = (
          watch_signal    * 4.0
        + emotional_onset * 2.0
        + right_dom_t0    * 0.02
        - dropoff_risk    * 0.04
        + mean_sustained  * 3.0
    )
    return {
        "watch_signal":    round(watch_signal, 4),
        "emotional_onset": round(emotional_onset, 4),
        "right_dom_t0":    round(right_dom_t0, 1),
        "dropoff_risk":    round(dropoff_risk, 1),
        "mean_sustained":  round(mean_sustained, 4),
        "viral_score":     round(viral, 3),
    }


if __name__ == "__main__":
    import tempfile
    import os
    from tribev2 import TribeModel

    hook_text = sys.argv[1]
    out_json  = sys.argv[2]

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(hook_text)
        tmp_path = f.name

    try:
        df = model.get_events_dataframe(text_path=tmp_path)
        preds, segments = model.predict(events=df)
    finally:
        os.unlink(tmp_path)

    seconds = compute_metrics(preds)
    viral   = compute_viral_score(seconds)

    result = {"hook": hook_text, "seconds": seconds, "viral": viral}

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {out_json}", flush=True)
