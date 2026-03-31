if __name__ == '__main__':
    import numpy as np
    from tribev2 import TribeModel

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    df = model.get_events_dataframe(text_path="hook.txt")
    preds, segments = model.predict(events=df)

    # preds shape: (5 timesteps, 20484 vertices)
    # 20484 = 10242 left hemisphere + 10242 right hemisphere (fsaverage5)
    n_vertices_per_hemi = preds.shape[1] // 2
    left = preds[:, :n_vertices_per_hemi]
    right = preds[:, n_vertices_per_hemi:]

    # Print basic stats per second
    print("\n=== BRAIN ACTIVATION PER SECOND ===")
    for i, seg in enumerate(segments):
        print(f"\nSecond {i} (t={seg.start:.1f}-{seg.start+seg.duration:.1f}s):")
        print(f"  Overall activation: mean={preds[i].mean():.4f}, max={preds[i].max():.4f}, min={preds[i].min():.4f}")
        print(f"  Left hemisphere:  mean={left[i].mean():.4f}, max={left[i].max():.4f}")
        print(f"  Right hemisphere: mean={right[i].mean():.4f}, max={right[i].max():.4f}")

    # Find top 100 most activated vertices per second
    print("\n=== PEAK ACTIVATION ANALYSIS ===")
    for i, seg in enumerate(segments):
        top_indices = np.argsort(preds[i])[-100:]
        top_mean = preds[i][top_indices].mean()
        top_left = sum(1 for idx in top_indices if idx < n_vertices_per_hemi)
        top_right = 100 - top_left
        print(f"Second {i}: top-100 mean={top_mean:.4f}, left={top_left}, right={top_right}")

    # Overall hook score
    print(f"\n=== HOOK SUMMARY ===")
    print(f"Total duration: {len(segments)} seconds")
    print(f"Peak second: {np.argmax([preds[i].max() for i in range(len(segments))])}")
    print(f"Strongest overall activation: {preds.max():.4f}")
    print(f"Weakest second (lowest mean): {np.argmin([preds[i].mean() for i in range(len(segments))])}")

    # Save raw data for later
    np.save("hook_preds.npy", preds)
    print("\nRaw predictions saved to hook_preds.npy")
