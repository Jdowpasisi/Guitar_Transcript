"""
src/ml/evaluate_fusion.py
P12: Fusion Model — Evaluation Script.

Evaluates the trained FusionModel under three conditions:
    1. Audio-only — video features zeroed (tests graceful degradation)
    2. Video-only — audio features zeroed (tests video pathway)
    3. Fused      — both modalities present (should win)

Also compares against the P6 greedy baseline and LSTM results.
Optionally tests under noisy audio conditions.

Run:
    python -m src.ml.evaluate_fusion
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import numpy as np

# ── resolve project root ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
for _candidate in [_HERE.parents[2], _HERE.parent, Path.cwd()]:
    if (_candidate / "src" / "config.py").exists():
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEVICE, MODELS_DIR, OUTPUTS_DIR,
    FUSION_MODEL_PATH, FUSION_EVAL_REPORT,
    P12_NUM_POSITIONS,
)
from src.ml.fusion_dataset import (
    FusionDataset, fusion_collate_fn, PAD_LABEL,
)
from src.ml.fusion_model import FusionModel
from src.ml.voicing_dataset import (
    VoicingDataset, position_from_index,
    OPEN_STRINGS, MAX_FRET, position_index,
)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation functions
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fusion_model(
    model: FusionModel,
    dataset: FusionDataset,
    device: str = "cpu",
    mode: str = "fused",
    audio_noise_std: float = 0.0,
) -> Tuple[float, dict]:
    """
    Evaluate the fusion model under a specific condition.

    Args:
        model:  trained FusionModel
        dataset: FusionDataset (test split)
        device: torch device
        mode:   "fused" | "audio_only" | "video_only"
        audio_noise_std: optional Gaussian noise added to audio features

    Returns:
        (tab_accuracy, per_string_breakdown)
    """
    model.eval()
    total_correct = 0
    total_notes = 0

    per_string_correct = [0] * 6
    per_string_total = [0] * 6

    with torch.no_grad():
        for idx in range(len(dataset)):
            audio, video, labels, length = dataset[idx]

            # Apply mode
            if mode == "audio_only":
                video = torch.zeros_like(video)
            elif mode == "video_only":
                audio = torch.zeros_like(audio)

            # Apply audio noise
            if audio_noise_std > 0 and mode != "video_only":
                noise = torch.randn_like(audio) * audio_noise_std
                audio = audio + noise

            # Unsqueeze for batch dim
            audio = audio.unsqueeze(0).to(device)       # (1, N, 56)
            video = video.unsqueeze(0).to(device)       # (1, N, 7)
            lengths = torch.tensor([length], dtype=torch.long, device=device)

            logits = model(audio, video, lengths)       # (1, N, 138)
            preds = logits[0].argmax(dim=-1)            # (N,)

            for i in range(length):
                pred_idx = preds[i].item()
                true_idx = labels[i].item()
                correct = int(pred_idx == true_idx)

                total_correct += correct
                total_notes += 1

                true_s, _ = position_from_index(true_idx)
                if 0 <= true_s < 6:
                    per_string_total[true_s] += 1
                    per_string_correct[true_s] += correct

    accuracy = total_correct / max(total_notes, 1)

    string_detail = {}
    for s in range(6):
        string_detail[f"string_{s}"] = {
            "accuracy": round(per_string_correct[s] / max(per_string_total[s], 1), 4),
            "n_notes": per_string_total[s],
        }

    return accuracy, string_detail


def evaluate_greedy_baseline(test_ds_voicing: VoicingDataset) -> float:
    """
    Evaluate the P6 greedy heuristic baseline on the voicing test set.
    Returns tab accuracy.
    """
    from src.ml.evaluate_voicing import greedy_voicing, evaluate_greedy
    return evaluate_greedy(test_ds_voicing)


def evaluate_lstm_baseline(device: str = "cpu") -> float | None:
    """
    Evaluate the P6 VoicingLSTM on the test set.
    Returns tab accuracy, or None if checkpoint not found.
    """
    lstm_path = str(MODELS_DIR / "voicing_lstm.pth")
    if not os.path.exists(lstm_path):
        return None

    try:
        from src.ml.models import VoicingLSTM
        from src.ml.evaluate_voicing import evaluate_lstm
        test_ds = VoicingDataset("test")
        model = VoicingLSTM.load(lstm_path, device=device)
        model.to(device)
        return evaluate_lstm(model, test_ds, device=device)
    except Exception as e:
        print(f"  ⚠ Could not evaluate LSTM: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate():
    print("=" * 60)
    print("P12 Fusion Model — Evaluation (Player 05 test split)")
    print("=" * 60)

    # ── Check for fusion model checkpoint ─────────────────────────────────
    checkpoint = str(FUSION_MODEL_PATH)
    if not os.path.exists(checkpoint):
        print(f"\n❌ Fusion model checkpoint not found at {checkpoint}.")
        print("   Train the model first:  python -m src.ml.train_fusion")
        return

    # ── Load fusion model ─────────────────────────────────────────────────
    print(f"\nLoading fusion model from {checkpoint} ...")
    model = FusionModel.load(checkpoint, device=DEVICE)
    model.to(DEVICE)
    print(f"  Parameters: {model.num_parameters:,}")

    # ── Load test dataset (with clean video for fair comparison) ───────────
    print("\nLoading test dataset ...")
    test_ds = FusionDataset("test", video_noise_std=0.0, video_dropout=0.0, seed=999)
    if len(test_ds) == 0:
        print("❌ No test recordings found. Check GuitarSet path.")
        return

    total_notes = sum(len(l) for l in test_ds.labels)
    print(f"  {len(test_ds)} recordings, {total_notes} notes total.")

    # ── Evaluate P6 baselines ─────────────────────────────────────────────
    print("\n── P6 Baselines ──")
    try:
        voicing_test_ds = VoicingDataset("test")
        greedy_acc = evaluate_greedy_baseline(voicing_test_ds)
        print(f"  Greedy Baseline Tab Accuracy: {greedy_acc:.1%}")
    except Exception as e:
        greedy_acc = None
        print(f"  ⚠ Greedy baseline failed: {e}")

    lstm_acc = evaluate_lstm_baseline(device=DEVICE)
    if lstm_acc is not None:
        print(f"  Bi-LSTM Tab Accuracy:         {lstm_acc:.1%}")
    else:
        print(f"  Bi-LSTM: checkpoint not found (skipping)")

    # ── Evaluate Fusion Model — 3 conditions ──────────────────────────────
    print("\n── P12 Fusion Model — 3-Condition Evaluation ──")

    # Condition 1: Audio-only
    print("\n  Evaluating audio-only mode ...")
    audio_only_acc, audio_only_detail = evaluate_fusion_model(
        model, test_ds, device=DEVICE, mode="audio_only",
    )
    print(f"  Audio-Only Tab Accuracy:      {audio_only_acc:.1%}")

    # Condition 2: Video-only
    print("  Evaluating video-only mode ...")
    video_only_acc, video_only_detail = evaluate_fusion_model(
        model, test_ds, device=DEVICE, mode="video_only",
    )
    print(f"  Video-Only Tab Accuracy:      {video_only_acc:.1%}")

    # Condition 3: Fused
    print("  Evaluating fused mode ...")
    fused_acc, fused_detail = evaluate_fusion_model(
        model, test_ds, device=DEVICE, mode="fused",
    )
    print(f"  Fused Tab Accuracy:           {fused_acc:.1%}")

    # ── Noisy audio test ──────────────────────────────────────────────────
    print("\n── Noisy Audio Test ──")
    noise_levels = [0.1, 0.3, 0.5]
    noisy_results = {}
    for noise_std in noise_levels:
        audio_noisy_acc, _ = evaluate_fusion_model(
            model, test_ds, device=DEVICE, mode="audio_only",
            audio_noise_std=noise_std,
        )
        fused_noisy_acc, _ = evaluate_fusion_model(
            model, test_ds, device=DEVICE, mode="fused",
            audio_noise_std=noise_std,
        )
        noisy_results[f"noise_{noise_std}"] = {
            "audio_only": round(audio_noisy_acc, 5),
            "fused": round(fused_noisy_acc, 5),
            "improvement": round(fused_noisy_acc - audio_noisy_acc, 5),
        }
        print(f"  Noise σ={noise_std:.1f}: "
              f"Audio-only={audio_noisy_acc:.1%}  "
              f"Fused={fused_noisy_acc:.1%}  "
              f"Δ={fused_noisy_acc - audio_noisy_acc:+.1%}")

    # ── Results table ─────────────────────────────────────────────────────
    print()
    print("═" * 55)
    print(f"  {'Model':<30s}  {'Tab Accuracy':>12s}")
    print("─" * 55)
    if greedy_acc is not None:
        print(f"  {'P6 Greedy Baseline':<30s}  {greedy_acc:>11.1%}")
    if lstm_acc is not None:
        print(f"  {'P6 Bi-LSTM':<30s}  {lstm_acc:>11.1%}")
    print(f"  {'P12 Audio-Only (fallback)':<30s}  {audio_only_acc:>11.1%}")
    print(f"  {'P12 Video-Only':<30s}  {video_only_acc:>11.1%}")
    print(f"  {'P12 Fused (audio+video)':<30s}  {fused_acc:>11.1%}")
    print("═" * 55)

    # ── Per-string breakdown (fused) ──────────────────────────────────────
    if fused_detail:
        print("\n  Per-string accuracy (Fused model):")
        note_names = ["E2", "A2", "D3", "G3", "B3", "E4"]
        for s in range(6):
            info = fused_detail.get(f"string_{s}", {})
            acc = info.get("accuracy", 0.0)
            n = info.get("n_notes", 0)
            print(f"    String {s} ({note_names[s]}):  {acc:.1%}  ({n} notes)")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(str(FUSION_EVAL_REPORT)) or ".", exist_ok=True)
    results = {
        "split": "test (Player 05)",
        "n_recordings": len(test_ds),
        "n_notes": int(total_notes),
        "model_params": model.num_parameters,
        "baselines": {
            "greedy_tab_acc": round(greedy_acc, 5) if greedy_acc is not None else None,
            "lstm_tab_acc": round(lstm_acc, 5) if lstm_acc is not None else None,
        },
        "fusion_results": {
            "audio_only_tab_acc": round(audio_only_acc, 5),
            "video_only_tab_acc": round(video_only_acc, 5),
            "fused_tab_acc": round(fused_acc, 5),
        },
        "noisy_audio_test": noisy_results,
        "fused_per_string": fused_detail,
        "audio_only_per_string": audio_only_detail,
        "video_only_per_string": video_only_detail,
    }

    with open(str(FUSION_EVAL_REPORT), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {FUSION_EVAL_REPORT}")


if __name__ == "__main__":
    evaluate()
