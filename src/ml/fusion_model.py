"""
src/ml/fusion_model.py
P12: Fusion Model — Cross-Attention Architecture.

A multimodal model that fuses audio features with video (finger tracking)
features using cross-attention to predict (string, fret) positions more
accurately than either modality alone.

Architecture:
    AudioEncoder  → audio tokens  (B, T, model_dim)
    VideoEncoder  → video tokens  (B, T, model_dim)
    Bi-LSTM       → sequential context on audio tokens
    Cross-Attention ×2 → audio attends to video
    Self-Attention     → final fusion
    Output Head        → Linear(model_dim → 138)

Graceful degradation: when video_available=0 for all notes, video tokens
are zero vectors → cross-attention effectively passes audio through →
model degrades to audio-only quality.

Run as __main__ for a quick forward-pass test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    P12_MODEL_DIM, P12_NUM_HEADS, P12_FF_DIM,
    P12_NUM_CROSS_ATTN, P12_LSTM_HIDDEN, P12_LSTM_LAYERS,
    P12_DROPOUT, P12_NUM_POSITIONS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Audio Encoder
# ══════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """
    Projects audio features (56d) to model dimension.

    Linear(56 → 128) → LayerNorm → GELU → Dropout → Linear(128 → model_dim)
    """

    def __init__(self, input_dim: int = AUDIO_FEATURE_DIM,
                 hidden_dim: int = 128,
                 output_dim: int = P12_MODEL_DIM,
                 dropout: float = P12_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, T, output_dim)"""
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Video Encoder
# ══════════════════════════════════════════════════════════════════════════════

class VideoEncoder(nn.Module):
    """
    Projects video features (7d) to model dimension.

    Linear(7 → 64) → LayerNorm → GELU → Dropout → Linear(64 → model_dim)
    """

    def __init__(self, input_dim: int = VIDEO_FEATURE_DIM,
                 hidden_dim: int = 64,
                 output_dim: int = P12_MODEL_DIM,
                 dropout: float = P12_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, T, output_dim)"""
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Attention Block
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: audio queries attend to video keys/values.
    Pre-norm architecture with residual connections.

    Audio (Q) attends to Video (K, V) → fused representation.
    When video is zero, attention weights are ~uniform → passes audio through.
    """

    def __init__(self, model_dim: int = P12_MODEL_DIM,
                 num_heads: int = P12_NUM_HEADS,
                 ff_dim: int = P12_FF_DIM,
                 dropout: float = P12_DROPOUT):
        super().__init__()

        # Cross-attention: Q from audio, K/V from video
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm_kv = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio: torch.Tensor,      # (B, T, model_dim) — queries
        video: torch.Tensor,      # (B, T, model_dim) — keys/values
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=pad
    ) -> torch.Tensor:
        """
        Returns: fused tensor (B, T, model_dim)
        """
        # Pre-norm cross-attention with residual
        q = self.norm_q(audio)
        kv = self.norm_kv(video)
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        x = audio + self.attn_dropout(attn_out)

        # Pre-norm feed-forward with residual
        ff_out = self.ff(self.norm_ff(x))
        x = x + ff_out

        return x


# ══════════════════════════════════════════════════════════════════════════════
# Self-Attention Block
# ══════════════════════════════════════════════════════════════════════════════

class SelfAttentionBlock(nn.Module):
    """
    Standard self-attention for processing fused features.
    Pre-norm architecture with residual connections.
    """

    def __init__(self, model_dim: int = P12_MODEL_DIM,
                 num_heads: int = P12_NUM_HEADS,
                 ff_dim: int = P12_FF_DIM,
                 dropout: float = P12_DROPOUT):
        super().__init__()

        self.norm_attn = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                                  # (B, T, model_dim)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, T) True=pad
    ) -> torch.Tensor:
        # Pre-norm self-attention with residual
        normed = self.norm_attn(x)
        attn_out, _ = self.self_attn(normed, normed, normed,
                                     key_padding_mask=key_padding_mask)
        x = x + self.attn_dropout(attn_out)

        # Pre-norm FF with residual
        ff_out = self.ff(self.norm_ff(x))
        x = x + ff_out

        return x


# ══════════════════════════════════════════════════════════════════════════════
# Fusion Model (the complete architecture)
# ══════════════════════════════════════════════════════════════════════════════

class FusionModel(nn.Module):
    """
    Cross-attention Fusion Model for multimodal (string, fret) prediction.

    Architecture:
        AudioEncoder(56 → model_dim)
        VideoEncoder(7 → model_dim)
        2-layer Bi-LSTM on audio tokens (sequential context)
        CrossAttention ×2 (audio attends to video)
        SelfAttention ×1 (final fusion)
        OutputHead(model_dim → 138)

    Input:
        audio_features : (B, T, 56)  — audio features per note
        video_features : (B, T, 7)   — video features per note
        lengths        : (B,)        — actual sequence lengths

    Output:
        logits : (B, T, 138)  — scores for each (string, fret) position
    """

    def __init__(
        self,
        audio_dim: int = AUDIO_FEATURE_DIM,
        video_dim: int = VIDEO_FEATURE_DIM,
        model_dim: int = P12_MODEL_DIM,
        num_heads: int = P12_NUM_HEADS,
        ff_dim: int = P12_FF_DIM,
        num_cross_attn: int = P12_NUM_CROSS_ATTN,
        lstm_hidden: int = P12_LSTM_HIDDEN,
        lstm_layers: int = P12_LSTM_LAYERS,
        dropout: float = P12_DROPOUT,
        num_positions: int = P12_NUM_POSITIONS,
    ):
        super().__init__()

        # Store hyperparams for save/load
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_cross_attn = num_cross_attn
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout
        self.num_positions = num_positions

        # Modality encoders
        self.audio_encoder = AudioEncoder(audio_dim, hidden_dim=128,
                                          output_dim=model_dim, dropout=dropout)
        self.video_encoder = VideoEncoder(video_dim, hidden_dim=64,
                                          output_dim=model_dim, dropout=dropout)

        # Sequential context encoder (Bi-LSTM on audio tokens)
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Project Bi-LSTM output back to model_dim
        self.lstm_proj = nn.Linear(lstm_hidden * 2, model_dim)

        # Cross-attention layers (audio queries attend to video keys/values)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_cross_attn)
        ])

        # Self-attention layer (process fused features)
        self.self_attention = SelfAttentionBlock(model_dim, num_heads, ff_dim, dropout)

        # Final layer norm
        self.final_norm = nn.LayerNorm(model_dim)

        # Output head
        self.output_head = nn.Linear(model_dim, num_positions)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers; orthogonal for LSTM weights."""
        for module in [self.audio_encoder, self.video_encoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        nn.init.xavier_uniform_(self.lstm_proj.weight)
        nn.init.zeros_(self.lstm_proj.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(
        self,
        audio_features: torch.Tensor,   # (B, T, audio_dim)
        video_features: torch.Tensor,   # (B, T, video_dim)
        lengths: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        """
        Returns: logits (B, T, num_positions)
        """
        B, T, _ = audio_features.shape

        # ── Encode modalities ─────────────────────────────────────────────
        audio_tokens = self.audio_encoder(audio_features)   # (B, T, model_dim)
        video_tokens = self.video_encoder(video_features)   # (B, T, model_dim)

        # ── Sequential context via Bi-LSTM ────────────────────────────────
        packed = pack_padded_sequence(
            audio_tokens, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=True,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Pad back to original T if needed
        if lstm_out.size(1) < T:
            pad_len = T - lstm_out.size(1)
            lstm_out = F.pad(lstm_out, (0, 0, 0, pad_len))

        audio_context = self.lstm_proj(lstm_out)   # (B, T, model_dim)

        # ── Build padding mask ────────────────────────────────────────────
        # True = padded position (to be ignored by attention)
        key_padding_mask = torch.arange(T, device=audio_features.device).unsqueeze(0) >= lengths.to(audio_features.device).unsqueeze(1)

        # ── Cross-attention (audio attends to video) ──────────────────────
        x = audio_context
        for cross_attn in self.cross_attention_layers:
            x = cross_attn(x, video_tokens, key_padding_mask=key_padding_mask)

        # ── Self-attention on fused features ──────────────────────────────
        x = self.self_attention(x, key_padding_mask=key_padding_mask)

        # ── Output head ───────────────────────────────────────────────────
        x = self.final_norm(x)
        logits = self.output_head(x)   # (B, T, 138)

        return logits

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str = "models/fusion_model.pth"):
        """Save model weights and hyperparams to a checkpoint file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "audio_dim": self.audio_dim,
            "video_dim": self.video_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_cross_attn": self.num_cross_attn,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout_rate,
            "num_positions": self.num_positions,
            "state_dict": self.state_dict(),
        }, path)
        print(f"[FusionModel] Saved checkpoint → {path}")

    @classmethod
    def load(cls, path: str = "models/fusion_model.pth",
             device: str = "cpu") -> "FusionModel":
        """Load a checkpoint saved with .save()."""
        ckpt = torch.load(path, map_location=device)
        model = cls(
            audio_dim=ckpt["audio_dim"],
            video_dim=ckpt["video_dim"],
            model_dim=ckpt["model_dim"],
            num_heads=ckpt["num_heads"],
            ff_dim=ckpt["ff_dim"],
            num_cross_attn=ckpt["num_cross_attn"],
            lstm_hidden=ckpt["lstm_hidden"],
            lstm_layers=ckpt["lstm_layers"],
            dropout=ckpt["dropout"],
            num_positions=ckpt["num_positions"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ── quick forward-pass test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FusionModel — forward-pass test")
    print("=" * 60)

    model = FusionModel()
    print(f"\nFusionModel (P12)")
    print(f"  Parameters : {model.num_parameters:,}")

    B, T = 4, 50
    audio = torch.randn(B, T, AUDIO_FEATURE_DIM)
    video = torch.randn(B, T, VIDEO_FEATURE_DIM)
    lengths = torch.tensor([50, 45, 30, 20])

    logits = model(audio, video, lengths)
    print(f"\n  Input  : audio={tuple(audio.shape)}, video={tuple(video.shape)}, "
          f"lengths={lengths.tolist()}")
    print(f"  Output : {tuple(logits.shape)}   (B, T, 138)")

    # Test audio-only mode (video zeroed)
    video_zero = torch.zeros_like(video)
    logits_audio_only = model(audio, video_zero, lengths)
    print(f"\n  Audio-only mode (video=0):")
    print(f"  Output : {tuple(logits_audio_only.shape)}   (should still work)")

    # Test video-only mode (audio zeroed)
    audio_zero = torch.zeros_like(audio)
    logits_video_only = model(audio_zero, video, lengths)
    print(f"\n  Video-only mode (audio=0):")
    print(f"  Output : {tuple(logits_video_only.shape)}   (should still work)")

    # Verify outputs differ
    diff_av = (logits - logits_audio_only).abs().mean().item()
    diff_vonly = (logits - logits_video_only).abs().mean().item()
    print(f"\n  Mean |fused - audio_only| : {diff_av:.4f}")
    print(f"  Mean |fused - video_only| : {diff_vonly:.4f}")

    print("\n✅ FusionModel forward-pass OK.")
