#!/usr/bin/env zsh
# run_hooks.sh
# Tests 5 rewritten hooks through TRIBE v2 and exports brain data for comparison.
# Usage: ./run_hooks.sh
# Output: hook_results/ directory with one JSON per hook + a summary comparison.

set -euo pipefail

RESULTS_DIR="hook_results"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Python runner: predicts + exports brain data for a given hook text
# ---------------------------------------------------------------------------
run_hook() {
    local hook_text="$1"
    local out_json="$2"
    local tmp_py
    tmp_py=$(mktemp /tmp/tribe_hook_XXXXXX.py)

    cat > "$tmp_py" <<'PYEOF'
import sys
import json
import numpy as np
import tempfile
import os
from tribev2 import TribeModel

if __name__ == '__main__':
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

    n_hemi = preds.shape[1] // 2

    output = {
        "hook": hook_text,
        "seconds": []
    }

    for i in range(preds.shape[0]):
        row = preds[i]
        top100_idx = np.argsort(row)[-100:]
        second = {
            "t": i,
            "mean":            round(float(row.mean()), 4),
            "max":             round(float(row.max()),  4),
            "min":             round(float(row.min()),  4),
            "left_mean":       round(float(row[:n_hemi].mean()), 4),
            "right_mean":      round(float(row[n_hemi:].mean()), 4),
            "top100_mean":     round(float(np.sort(row)[-100:].mean()), 4),
            "top100_left_pct": int(sum(1 for idx in top100_idx if idx < n_hemi)),
        }
        output["seconds"].append(second)

    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {out_json}")
PYEOF

    python "$tmp_py" "$hook_text" "$out_json"
    rm -f "$tmp_py"
}

# ---------------------------------------------------------------------------
# Hook 1: WATCH SIGNAL — sustained activation through t=1-2
# ---------------------------------------------------------------------------
HOOK_TEXT="The most-banned app in Silicon Valley just shut down. Here's what they never told you."
HOOK_LABEL="watch_signal"
echo "[1/5] Mechanic: $HOOK_LABEL"
echo "      Hook: $HOOK_TEXT"
run_hook "$HOOK_TEXT" "${RESULTS_DIR}/hook_1_${HOOK_LABEL}.json"
echo ""

# ---------------------------------------------------------------------------
# Hook 2: SELF-RELEVANCE — medial prefrontal activation via "You"
# ---------------------------------------------------------------------------
HOOK_TEXT="You watched their videos. You shared them. You had no idea the founder was lying to everyone."
HOOK_LABEL="self_relevance"
echo "[2/5] Mechanic: $HOOK_LABEL"
echo "      Hook: $HOOK_TEXT"
run_hook "$HOOK_TEXT" "${RESULTS_DIR}/hook_2_${HOOK_LABEL}.json"
echo ""

# ---------------------------------------------------------------------------
# Hook 3: EMOTIONAL SALIENCE — right hemisphere dominance via human distress
# ---------------------------------------------------------------------------
HOOK_TEXT="The founder cried on his last livestream. A billion people had watched him. The money was already gone."
HOOK_LABEL="emotional_salience"
echo "[3/5] Mechanic: $HOOK_LABEL"
echo "      Hook: $HOOK_TEXT"
run_hook "$HOOK_TEXT" "${RESULTS_DIR}/hook_3_${HOOK_LABEL}.json"
echo ""

# ---------------------------------------------------------------------------
# Hook 4: SHARE SIGNAL — bilateral activation via social network simulation
# ---------------------------------------------------------------------------
HOOK_TEXT="Every person you know watched Cluely. None of you knew the company was collapsing. Now it's gone."
HOOK_LABEL="share_signal"
echo "[4/5] Mechanic: $HOOK_LABEL"
echo "      Hook: $HOOK_TEXT"
run_hook "$HOOK_TEXT" "${RESULTS_DIR}/hook_4_${HOOK_LABEL}.json"
echo ""

# ---------------------------------------------------------------------------
# Hook 5: DROP-OFF PREVENTION — number escalation + unresolvable paradox
# ---------------------------------------------------------------------------
HOOK_TEXT='$20 million raised. One billion views. Three federal complaints. The founder says he'"'"'d do it again.'
HOOK_LABEL="dropoff_prevention"
echo "[5/5] Mechanic: $HOOK_LABEL"
echo "      Hook: $HOOK_TEXT"
run_hook "$HOOK_TEXT" "${RESULTS_DIR}/hook_5_${HOOK_LABEL}.json"
echo ""

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------
echo "Generating comparison summary..."

TMP_SUMMARY=$(mktemp /tmp/tribe_summary_XXXXXX.py)
cat > "$TMP_SUMMARY" <<'PYEOF'
import sys
import json
import glob
import os

results_dir = sys.argv[1]

files = []
baseline = "hook_brain_data.json"
if os.path.exists(baseline):
    files.append(("BASELINE (original)", baseline))

for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    label = os.path.basename(path).replace(".json", "").replace("hook_", "").replace("_", " ", 1)
    files.append((label, path))

def load(path):
    with open(path) as f:
        return json.load(f)

def score(data):
    secs = data["seconds"]
    t0 = secs[0] if len(secs) > 0 else {}
    t1 = secs[1] if len(secs) > 1 else {}
    t3 = secs[3] if len(secs) > 3 else {}

    watch_signal    = t0.get("mean", 0) + t1.get("mean", 0)
    emotional_onset = t0.get("top100_mean", 0)
    right_dom_t0    = 100 - t0.get("top100_left_pct", 50)
    dropoff_risk    = t3.get("top100_left_pct", 50)
    mean_sustained  = sum(s.get("mean", 0) for s in secs) / len(secs) if secs else 0

    viral = (
        watch_signal    * 4.0 +
        emotional_onset * 2.0 +
        right_dom_t0    * 0.02 -
        dropoff_risk    * 0.04 +
        mean_sustained  * 3.0
    )
    return {
        "watch_signal":    round(watch_signal, 4),
        "emotional_onset": round(emotional_onset, 4),
        "right_dom_t0":    round(right_dom_t0, 1),
        "dropoff_risk":    round(dropoff_risk, 1),
        "mean_sustained":  round(mean_sustained, 4),
        "viral_score":     round(viral, 3),
    }

col_w = 24
print("")
print("=" * 100)
print(f"{'Hook':<{col_w}}  {'Watch':<8} {'Emot.Onset':<12} {'RightDom%':<11} {'DropRisk%':<11} {'Sustained':<11} {'ViralScore'}")
print("-" * 100)

for label, path in files:
    data = load(path)
    s = score(data)
    print(
        f"{label:<{col_w}}  "
        f"{s['watch_signal']:<8} "
        f"{s['emotional_onset']:<12} "
        f"{s['right_dom_t0']:<11} "
        f"{s['dropoff_risk']:<11} "
        f"{s['mean_sustained']:<11} "
        f"{s['viral_score']}"
    )

print("=" * 100)
print("")
print("Metrics guide:")
print("  Watch signal    : t0.mean + t1.mean  (higher = viewer stays)")
print("  Emot. onset     : t0.top100_mean     (higher = stronger hotspot firing)")
print("  RightDom% t0    : 100 - top100_left_pct at t=0  (higher = more emotional)")
print("  DropOff risk%   : top100_left_pct at t=3        (lower = better, <50 = right still leading)")
print("  Mean sustained  : avg mean across all seconds   (higher = sustained engagement)")
print("  Viral score     : composite heuristic (higher = better)")
print("")
PYEOF

python "$TMP_SUMMARY" "$RESULTS_DIR"
rm -f "$TMP_SUMMARY"

echo "Done. Results saved to: $RESULTS_DIR/"
echo "Individual JSONs: $(ls ${RESULTS_DIR}/*.json | wc -l | tr -d ' ') files"
