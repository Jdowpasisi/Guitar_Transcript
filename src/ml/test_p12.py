"""
src/ml/test_p12.py
P12: Fusion Model — Smoke Test.

Validates all P12 components without requiring real data or trained models:
    1. FusionModel forward pass with random tensors
    2. Audio-only mode (video features zeroed)
    3. Video-only mode (audio features zeroed)
    4. CrossAttentionBlock shape validation
    5. Save/load checkpoint round-trip
    6. FusionDataset collate function
    7. Gradient flow test
    8. Parameter count verification

Run:
    python -m src.ml.test_p12
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
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
    AUDIO_FEATURE_DIM, VIDEO_FEATURE_DIM,
    P12_MODEL_DIM, P12_NUM_POSITIONS,
)


def run_smoke_test():
    """Run all P12 component smoke tests."""
    print("=" * 60)
    print("P12 Fusion Model — Smoke Test")
    print("=" * 60)

    passed = 0
    failed = 0

    # ── Test 1: FusionModel forward pass ──────────────────────────────────
    print("\n[1] FusionModel forward pass ...")
    try:
        from src.ml.fusion_model import FusionModel

        model = FusionModel()
        B, T = 4, 30
        audio = torch.randn(B, T, AUDIO_FEATURE_DIM)
        video = torch.randn(B, T, VIDEO_FEATURE_DIM)
        lengths = torch.tensor([30, 25, 15, 10])

        logits = model(audio, video, lengths)
        assert logits.shape == (B, T, P12_NUM_POSITIONS), \
            f"Expected ({B}, {T}, {P12_NUM_POSITIONS}), got {logits.shape}"
        print(f"    ✅ Output shape: {tuple(logits.shape)}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 2: Audio-only mode ───────────────────────────────────────────
    print("\n[2] Audio-only mode (video zeroed) ...")
    try:
        video_zero = torch.zeros(B, T, VIDEO_FEATURE_DIM)
        logits_audio = model(audio, video_zero, lengths)
        assert logits_audio.shape == (B, T, P12_NUM_POSITIONS)

        # Outputs should differ from fused mode
        diff = (logits - logits_audio).abs().mean().item()
        assert diff > 1e-6, f"Outputs should differ (diff={diff})"
        print(f"    ✅ Audio-only works, diff from fused: {diff:.4f}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 3: Video-only mode ───────────────────────────────────────────
    print("\n[3] Video-only mode (audio zeroed) ...")
    try:
        audio_zero = torch.zeros(B, T, AUDIO_FEATURE_DIM)
        logits_video = model(audio_zero, video, lengths)
        assert logits_video.shape == (B, T, P12_NUM_POSITIONS)

        diff = (logits - logits_video).abs().mean().item()
        assert diff > 1e-6, f"Outputs should differ (diff={diff})"
        print(f"    ✅ Video-only works, diff from fused: {diff:.4f}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 4: CrossAttentionBlock shape ─────────────────────────────────
    print("\n[4] CrossAttentionBlock shape validation ...")
    try:
        from src.ml.fusion_model import CrossAttentionBlock

        block = CrossAttentionBlock(model_dim=P12_MODEL_DIM)
        q = torch.randn(2, 20, P12_MODEL_DIM)
        kv = torch.randn(2, 20, P12_MODEL_DIM)
        mask = torch.tensor([[False]*15 + [True]*5,
                             [False]*10 + [True]*10])

        out = block(q, kv, key_padding_mask=mask)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"
        print(f"    ✅ Output shape: {tuple(out.shape)}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 5: SelfAttentionBlock shape ──────────────────────────────────
    print("\n[5] SelfAttentionBlock shape validation ...")
    try:
        from src.ml.fusion_model import SelfAttentionBlock

        sa_block = SelfAttentionBlock(model_dim=P12_MODEL_DIM)
        x = torch.randn(2, 20, P12_MODEL_DIM)
        out = sa_block(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        print(f"    ✅ Output shape: {tuple(out.shape)}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 6: Save/Load checkpoint round-trip ───────────────────────────
    print("\n[6] Save/Load checkpoint round-trip ...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_fusion.pth")
            model.eval()  # Set to eval mode to disable dropout
            model.save(ckpt_path)

            # Re-run forward pass in eval mode for consistent comparison
            with torch.no_grad():
                logits_eval = model(audio, video, lengths)

            model2 = FusionModel.load(ckpt_path)
            model2.eval()
            with torch.no_grad():
                logits2 = model2(audio, video, lengths)

            diff = (logits_eval - logits2).abs().max().item()
            assert diff < 1e-5, f"Loaded model differs: max diff = {diff}"
            print(f"    ✅ Round-trip OK (max diff: {diff:.2e})")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 7: Collate function ──────────────────────────────────────────
    print("\n[7] Collate function padding ...")
    try:
        from src.ml.fusion_dataset import fusion_collate_fn, PAD_LABEL

        # Simulate a mini-batch with different lengths
        batch = [
            (torch.randn(20, AUDIO_FEATURE_DIM), torch.randn(20, VIDEO_FEATURE_DIM),
             torch.randint(0, 138, (20,)), 20),
            (torch.randn(10, AUDIO_FEATURE_DIM), torch.randn(10, VIDEO_FEATURE_DIM),
             torch.randint(0, 138, (10,)), 10),
            (torch.randn(15, AUDIO_FEATURE_DIM), torch.randn(15, VIDEO_FEATURE_DIM),
             torch.randint(0, 138, (15,)), 15),
        ]

        audio_p, video_p, labels_p, lens = fusion_collate_fn(batch)

        assert audio_p.shape == (3, 20, AUDIO_FEATURE_DIM), \
            f"Audio shape: {audio_p.shape}"
        assert video_p.shape == (3, 20, VIDEO_FEATURE_DIM), \
            f"Video shape: {video_p.shape}"
        assert labels_p.shape == (3, 20), \
            f"Labels shape: {labels_p.shape}"
        assert lens.tolist() == [20, 15, 10], \
            f"Lengths (sorted desc): {lens.tolist()}"

        # Check padding
        assert (labels_p[2, 10:] == PAD_LABEL).all(), \
            "Shortest sequence should be padded with PAD_LABEL"

        print(f"    ✅ Padded shapes: audio={tuple(audio_p.shape)}, "
              f"labels={tuple(labels_p.shape)}, lens={lens.tolist()}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 8: Gradient flow ─────────────────────────────────────────────
    print("\n[8] Gradient flow test ...")
    try:
        model.train()
        audio_g = torch.randn(2, 10, AUDIO_FEATURE_DIM, requires_grad=True)
        video_g = torch.randn(2, 10, VIDEO_FEATURE_DIM, requires_grad=True)
        lens_g = torch.tensor([10, 8])

        logits_g = model(audio_g, video_g, lens_g)
        loss = logits_g.sum()
        loss.backward()

        # Check gradients exist
        has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "Not all parameters received gradients"

        # Check audio input grad exists
        assert audio_g.grad is not None, "Audio input has no gradient"
        assert video_g.grad is not None, "Video input has no gradient"

        print(f"    ✅ All parameters received gradients")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 9: Parameter count ───────────────────────────────────────────
    print("\n[9] Parameter count verification ...")
    try:
        n_params = model.num_parameters
        assert 100_000 < n_params < 5_000_000, \
            f"Parameter count {n_params:,} seems unreasonable"
        print(f"    ✅ Parameters: {n_params:,} (reasonable range)")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Test 10: Encoder modules ──────────────────────────────────────────
    print("\n[10] AudioEncoder + VideoEncoder standalone ...")
    try:
        from src.ml.fusion_model import AudioEncoder, VideoEncoder

        ae = AudioEncoder()
        ve = VideoEncoder()

        a_out = ae(torch.randn(2, 10, AUDIO_FEATURE_DIM))
        v_out = ve(torch.randn(2, 10, VIDEO_FEATURE_DIM))

        assert a_out.shape == (2, 10, P12_MODEL_DIM), f"AudioEncoder: {a_out.shape}"
        assert v_out.shape == (2, 10, P12_MODEL_DIM), f"VideoEncoder: {v_out.shape}"

        print(f"    ✅ AudioEncoder: {tuple(a_out.shape)}, "
              f"VideoEncoder: {tuple(v_out.shape)}")
        passed += 1
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        failed += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"✅ All {total} tests passed!")
    else:
        print(f"❌ {failed}/{total} tests failed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
