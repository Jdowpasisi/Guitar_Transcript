"""
evaluate_chord.py
-----------------
Evaluates the trained ChordCNN on the held-out test set.

Outputs:
  • Test accuracy and per-class F1 scores
  • Confusion matrix saved as outputs/confusion_matrix.png
  • Madmom baseline comparison (if madmom is installed)
  • Full report saved to outputs/eval_report.json

Run:  python -m src.ml.evaluate_chord
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader

from src.ml.models import load_model
from src.ml.train_chord import ChordDataset, DEVICE

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        ConfusionMatrixDisplay,
        f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not installed — skipping confusion matrix.")

try:
    import madmom
    HAS_MADMOM = True
except ImportError:
    HAS_MADMOM = False

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MODEL_PATH     = "models/chord_cnn.pth"
LABEL_MAP_FILE = Path("data/processed/chord_dataset/label_map.json")
OUT_DIR        = Path("outputs")
BATCH_SIZE     = 128


# ─────────────────────────────────────────────────────────
# INFERENCE PASS
# ─────────────────────────────────────────────────────────
def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the model over the entire test loader.
    Returns (all_preds, all_labels) as numpy arrays.
    """
    model.eval()
    preds_list  = []
    labels_list = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            preds   = logits.argmax(dim=-1).cpu().numpy()
            preds_list.append(preds)
            labels_list.append(y_batch.numpy())

    return np.concatenate(preds_list), np.concatenate(labels_list)


# ─────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    class_names: list[str],
    out_path:  str = "outputs/confusion_matrix.png",
):
    """
    Saves a normalised confusion matrix as a PNG.

    Reading the confusion matrix:
      • Diagonal = correctly classified chords (bright = good)
      • Off-diagonal = confusions
      • "Perfect Fifths" confusion (C:maj ↔ G:maj) is musically expected
      • "Relative minor" confusion (C:maj ↔ A:min) is also expected —
        they share 2/3 notes. If you see this, your model is musically aware!
    """
    if not HAS_SKLEARN:
        print("sklearn not available — skipping confusion matrix plot.")
        return

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    # Clamp to ≤50 classes for readability
    n = min(len(class_names), 50)
    cm_plot = cm[:n, :n]
    names   = class_names[:n]

    fig_size = max(10, n // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    disp     = ConfusionMatrixDisplay(confusion_matrix=cm_plot, display_labels=names)
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        xticks_rotation=90,
        values_format=".2f",
    )
    ax.set_title("Chord CNN — Normalised Confusion Matrix (Test Set)", fontsize=13)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved → {out_path}")


# ─────────────────────────────────────────────────────────
# MUSICAL ANALYSIS: COMMON CONFUSIONS
# ─────────────────────────────────────────────────────────
def analyse_confusions(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    label_map:   dict[str, int],
    top_n:       int = 10,
) -> list[dict]:
    """
    Returns the top-N most common misclassifications with musical context.
    Useful for understanding if confusions are "musically reasonable."
    """
    inv_map = {v: k for k, v in label_map.items()}
    confused = []

    for true_cls in np.unique(y_true):
        mask    = y_true == true_cls
        preds   = y_pred[mask]
        for pred_cls in np.unique(preds):
            if pred_cls == true_cls:
                continue
            count   = np.sum(preds == pred_cls)
            total   = mask.sum()
            confused.append({
                "true":  inv_map.get(int(true_cls), "?"),
                "pred":  inv_map.get(int(pred_cls), "?"),
                "count": int(count),
                "rate":  round(float(count) / total, 3),
            })

    confused.sort(key=lambda x: x["count"], reverse=True)
    return confused[:top_n]


# ─────────────────────────────────────────────────────────
# MADMOM BASELINE
# ─────────────────────────────────────────────────────────
def run_madmom_baseline(test_audio_paths: list[str]) -> float | None:
    """
    Runs madmom's DeepChromaChordRecognition on the test audio files
    and returns its accuracy against the label map.

    This gives you a published baseline to compare against:
    "My CNN: 72% vs. Madmom: 68%" is a real headline result.

    Requires:  pip install madmom
    """
    if not HAS_MADMOM:
        print("ℹ️  madmom not installed — skipping baseline.")
        print("   Install with:  pip install madmom")
        return None

    try:
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        proc = DeepChromaChordRecognitionProcessor()

        correct = 0
        total   = 0
        for path in test_audio_paths[:50]:   # cap at 50 for speed
            try:
                result = proc(path)
                # result is a list of (start, end, chord_label)
                # For a quick accuracy estimate we check if any chord was found
                if result:
                    total += 1
                    # NOTE: proper evaluation requires matching timestamps
                    # This is a placeholder — expand with timestamp alignment
            except Exception:
                continue

        if total == 0:
            return None

        print(f"   Madmom processed {total} files (accuracy requires timestamp alignment)")
        return None  # Full baseline requires timestamp-aligned comparison

    except Exception as e:
        print(f"⚠️  Madmom baseline failed: {e}")
        return None


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def evaluate():
    print("=" * 55)
    print("  GuitarAI — Chord CNN Evaluation")
    print(f"  Device : {DEVICE.upper()}")
    print("=" * 55)

    # ── Load label map ────────────────────────────────────
    with open(LABEL_MAP_FILE) as f:
        label_map = json.load(f)
    inv_map     = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]

    # ── Load model ────────────────────────────────────────
    model = load_model(MODEL_PATH).to(DEVICE)

    # ── Test DataLoader ───────────────────────────────────
    test_ds = ChordDataset("test", augment=False)
    print(f"\n📦 Test samples : {len(test_ds):,}")

    test_loader = DataLoader(
        test_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
    )

    # ── Run inference ─────────────────────────────────────
    print("\n🔍 Running inference on test set...")
    y_pred, y_true = predict_all(model, test_loader, DEVICE)

    # ── Core metrics ──────────────────────────────────────
    accuracy = 100.0 * (y_pred == y_true).mean()
    print(f"\n{'─'*40}")
    print(f"  Test Accuracy : {accuracy:.2f}%")

    if HAS_SKLEARN:
        macro_f1 = f1_score(y_true, y_pred, average="macro",  zero_division=0)
        wt_f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"  Macro F1      : {macro_f1:.4f}")
        print(f"  Weighted F1   : {wt_f1:.4f}")
        print(f"{'─'*40}")

        # Only report on classes that actually appear in this test set.
        # If the model never predicts a class, sklearn's class count won't
        # match len(class_names) and classification_report crashes.
        present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        present_names  = [inv_map.get(i, f"cls_{i}") for i in present_labels]

        print("\n📋 Per-class classification report:")
        print(classification_report(
            y_true, y_pred,
            labels        = present_labels,
            target_names  = present_names,
            zero_division = 0,
        ))

    # ── Confusion matrix ──────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names)

    # ── Musical confusion analysis ────────────────────────
    print("\n🎵 Top misclassifications:")
    confusions = analyse_confusions(y_true, y_pred, label_map, top_n=10)
    for c in confusions:
        print(f"   {c['true']:12s} → {c['pred']:12s}  "
              f"({c['count']:4d} times, {c['rate']*100:.1f}%)")

    # ── Madmom baseline ───────────────────────────────────
    print("\n🎸 Madmom baseline:")
    run_madmom_baseline([])   # pass test audio paths for full baseline

    # ── Save report ───────────────────────────────────────
    report = {
        "test_accuracy_pct": round(float(accuracy), 3),
        "num_classes":       len(label_map),
        "num_test_samples":  int(len(test_ds)),
        "top_confusions":    confusions,
    }
    if HAS_SKLEARN:
        report["macro_f1"]    = round(float(macro_f1), 4)
        report["weighted_f1"] = round(float(wt_f1), 4)

    report_path = OUT_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Evaluation report saved → {report_path}")


if __name__ == "__main__":
    evaluate()