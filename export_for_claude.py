if __name__ == '__main__':
    import numpy as np
    import json

    preds = np.load("hook_preds.npy")
    n_hemi = preds.shape[1] // 2

    output = {
        "hook": open("hook.txt").read().strip(),
        "seconds": []
    }

    for i in range(preds.shape[0]):
        second = {
            "t": i,
            "mean": round(float(preds[i].mean()), 4),
            "max": round(float(preds[i].max()), 4),
            "min": round(float(preds[i].min()), 4),
            "left_mean": round(float(preds[i, :n_hemi].mean()), 4),
            "right_mean": round(float(preds[i, n_hemi:].mean()), 4),
            "top100_mean": round(float(np.sort(preds[i])[-100:].mean()), 4),
            "top100_left_pct": int(sum(1 for idx in np.argsort(preds[i])[-100:] if idx < n_hemi)),
        }
        output["seconds"].append(second)

    with open("hook_brain_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved to hook_brain_data.json")
    print(json.dumps(output, indent=2))
