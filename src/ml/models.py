"""
src/ml/models.py  (P4 + P6 combined)
─────────────────────────────────────
Contains:
  • ConvBlock, ChordCNN       — P4: chord classification
  • VoicingLSTM               — P6: (string, fret) prediction from MIDI sequences

P10's ChordShapeCNN lives in src/vision/chord_shape_cnn.py but is
imported here for the unified model summary (run as __main__).

Run as __main__ to print model summaries for all architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ══════════════════════════════════════════════════════════════════════════════
# P4 — Chord Classifier CNN
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # 'same' padding keeps spatial dims
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class ChordCNN(nn.Module):
    """
    Input:  (batch, 1, 84, 87)   — 1 channel, 84 freq bins, 87 time frames
    Output: (batch, num_classes) — raw logits (use CrossEntropyLoss)

    Architecture:
      Block 1: 1→32  channels, MaxPool  →  (32, 42, 43)
      Block 2: 32→64 channels, MaxPool  →  (64, 21, 21)
      Block 3: 64→128 channels, no pool →  (128, 21, 21)
      Global Avg Pool                   →  (128, 1, 1)
      Flatten                           →  (128,)
      FC1: 128→256 + ReLU + Dropout(0.5)
      FC2: 256→num_classes
    """

    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.block1 = ConvBlock(1,   32,  pool=True)
        self.block2 = ConvBlock(32,  64,  pool=True)
        self.block3 = ConvBlock(64,  128, pool=False)
        self.gap     = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(128, 256)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)                       # (B, 32, 42, 43)
        x = self.block2(x)                       # (B, 64, 21, 21)
        x = self.block3(x)                       # (B, 128, 21, 21)
        x = self.gap(x)                          # (B, 128, 1, 1)
        x = self.flatten(x)                      # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))    # (B, 256)
        x = self.fc2(x)                          # (B, num_classes)
        return x

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=-1)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory + Save/Load
def build_model(num_classes, dropout=0.5):
    return ChordCNN(num_classes=num_classes, dropout=dropout)

def save_model(model, path="models/chord_cnn.pth"):
    torch.save({"num_classes": model.num_classes, "state_dict": model.state_dict()}, path)

def load_model(path="models/chord_cnn.pth"):
    checkpoint = torch.load(path, map_location="cpu")
    model = ChordCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# P6 — Voicing LSTM
# ══════════════════════════════════════════════════════════════════════════════

# Fretboard constants (must match voicing_dataset.py)
_NUM_STRINGS   = 6
_MAX_FRET      = 22
_NUM_POSITIONS = _NUM_STRINGS * (_MAX_FRET + 1)   # 138


class VoicingLSTM(nn.Module):
    """
    Bidirectional LSTM that predicts the (string, fret) position for each
    note in a MIDI sequence.

    Input per timestep (teacher-forced during training):
        midi_pitch    — integer 0..127, passed through an embedding layer
        prev_position — flat (string, fret) index 0..137, encoded as one-hot
                        and projected via a linear layer
        delta_t       — scalar inter-note timing (seconds)

    Architecture:
        MIDI Embedding  : 128 vocab → 64 dim
        Prev-pos linear : 138 → 32
        Δt scalar       : 1 dim (concatenated as-is)
        Total input dim : 64 + 32 + 1 = 97

        2-layer Bidirectional LSTM (128 units each direction, dropout=0.3)
        → output dim = 128 × 2 = 256

        Linear head: 256 → 138   (6 strings × 23 frets)
        Loss: CrossEntropyLoss (applied externally)

    Output:
        logits tensor of shape (B, T, 138) — raw scores for each position.

    Usage (inference):
        logits = model(midi_seq, prev_positions, delta_t, lengths)
        preds  = logits.argmax(dim=-1)   # (B, T)

    Usage (training):
        Teacher-forced: prev_positions comes from ground truth, not model output.
    """

    MIDI_VOCAB     = 128
    MIDI_EMB_DIM   = 64
    PREV_POS_DIM   = 32
    DT_DIM         = 1
    INPUT_DIM      = MIDI_EMB_DIM + PREV_POS_DIM + DT_DIM   # 97
    LSTM_HIDDEN    = 128
    LSTM_LAYERS    = 2
    LSTM_DROPOUT   = 0.3
    OUTPUT_DIM     = _NUM_POSITIONS   # 138

    def __init__(
        self,
        midi_vocab:    int   = MIDI_VOCAB,
        midi_emb_dim:  int   = MIDI_EMB_DIM,
        prev_pos_dim:  int   = PREV_POS_DIM,
        lstm_hidden:   int   = LSTM_HIDDEN,
        lstm_layers:   int   = LSTM_LAYERS,
        lstm_dropout:  float = LSTM_DROPOUT,
        num_positions: int   = OUTPUT_DIM,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.lstm_hidden   = lstm_hidden
        self.lstm_layers   = lstm_layers

        input_dim = midi_emb_dim + prev_pos_dim + 1   # +1 for Δt scalar

        # MIDI pitch → dense vector
        self.midi_embedding = nn.Embedding(
            num_embeddings=midi_vocab,
            embedding_dim=midi_emb_dim,
            padding_idx=0,   # pitch 0 is used for padded steps
        )

        # Previous (string, fret) position → dense vector
        # Input: one-hot of size num_positions  →  projected to prev_pos_dim
        self.prev_pos_proj = nn.Linear(num_positions, prev_pos_dim, bias=True)

        # 2-layer Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Output head: (B, T, 2*hidden) → (B, T, 138)
        self.output_head = nn.Linear(lstm_hidden * 2, num_positions)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers; orthogonal for LSTM weights."""
        nn.init.xavier_uniform_(self.prev_pos_proj.weight)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (helps with long-range dependencies)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        midi_pitches:    torch.Tensor,   # (B, T)  long
        prev_positions:  torch.Tensor,   # (B, T)  long  — teacher-forced labels
        delta_t:         torch.Tensor,   # (B, T)  float
        lengths:         torch.Tensor,   # (B,)    long  — actual seq lengths
    ) -> torch.Tensor:
        """
        Returns:
            logits : FloatTensor (B, T, 138) — unnormalised scores
        """
        B, T = midi_pitches.shape

        # ── MIDI embedding ────────────────────────────────────────────────────
        midi_clamped = midi_pitches.clamp(0, self.midi_embedding.num_embeddings - 1)
        midi_emb = self.midi_embedding(midi_clamped)          # (B, T, 64)

        # ── Previous-position one-hot projection ──────────────────────────────
        prev_clamped = prev_positions.clamp(0, self.num_positions - 1)
        prev_onehot  = F.one_hot(prev_clamped, num_classes=self.num_positions).float()
        prev_proj    = self.prev_pos_proj(prev_onehot)        # (B, T, 32)

        # ── Δt feature ────────────────────────────────────────────────────────
        dt = delta_t.unsqueeze(-1)                            # (B, T, 1)

        # ── Concatenate features ──────────────────────────────────────────────
        x = torch.cat([midi_emb, prev_proj, dt], dim=-1)     # (B, T, 97)

        # ── Pack → LSTM → Unpack ──────────────────────────────────────────────
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _   = pad_packed_sequence(packed_out, batch_first=True)
        # lstm_out : (B, T_actual, 256)  — T_actual may differ from T if batch
        # Pad back to original T so shapes are consistent within the batch
        if lstm_out.size(1) < T:
            pad_len = T - lstm_out.size(1)
            lstm_out = F.pad(lstm_out, (0, 0, 0, pad_len))

        # ── Output head ───────────────────────────────────────────────────────
        logits = self.output_head(lstm_out)                   # (B, T, 138)
        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Save / Load helpers ───────────────────────────────────────────────────

    def save(self, path: str = "models/voicing_lstm.pth"):
        """Save model weights and hyperparams to a checkpoint file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "num_positions": self.num_positions,
            "lstm_hidden":   self.lstm_hidden,
            "lstm_layers":   self.lstm_layers,
            "state_dict":    self.state_dict(),
        }, path)
        print(f"[VoicingLSTM] Saved checkpoint → {path}")

    @classmethod
    def load(cls, path: str = "models/voicing_lstm.pth",
             device: str = "cpu") -> "VoicingLSTM":
        """Load a checkpoint saved with .save()."""
        ckpt  = torch.load(path, map_location=device)
        model = cls(
            num_positions=ckpt["num_positions"],
            lstm_hidden=  ckpt["lstm_hidden"],
            lstm_layers=  ckpt["lstm_layers"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ── quick model summary ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model summaries")
    print("=" * 60)

    cnn = ChordCNN(num_classes=51)
    print(f"\nChordCNN (P4)")
    print(f"  Parameters : {cnn.num_parameters:,}")
    x = torch.randn(2, 1, 84, 87)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(cnn(x).shape)}")

    print()

    lstm = VoicingLSTM()
    print(f"VoicingLSTM (P6)")
    print(f"  Parameters : {lstm.num_parameters:,}")
    B, T = 4, 50
    midi  = torch.randint(40, 87, (B, T))
    prev  = torch.randint(0, 138, (B, T))
    dt    = torch.rand(B, T)
    lens  = torch.tensor([50, 45, 30, 20])
    out   = lstm(midi, prev, dt, lens)
    print(f"  Input  : midi={tuple(midi.shape)}, prev={tuple(prev.shape)}, "
          f"dt={tuple(dt.shape)}, lengths={lens.tolist()}")
    print(f"  Output : {tuple(out.shape)}   (B, T, 138)")

    # ── P10: ChordShapeCNN ────────────────────────────────────────────────
    print()
    try:
        from src.vision.chord_shape_cnn import ChordShapeCNN
        from src.config import CHORD_INPUT_H, CHORD_INPUT_W, NUM_CHORD_SHAPES

        chord_shape = ChordShapeCNN(num_classes=NUM_CHORD_SHAPES)
        print(f"ChordShapeCNN (P10)")
        print(f"  Parameters : {chord_shape.num_parameters:,}")
        x_cs = torch.randn(2, 3, CHORD_INPUT_H, CHORD_INPUT_W)
        out_cs = chord_shape(x_cs)
        print(f"  Input  : {tuple(x_cs.shape)}  (B, C, H={CHORD_INPUT_H}, W={CHORD_INPUT_W})")
        print(f"  Output : {tuple(out_cs.shape)}   (B, {NUM_CHORD_SHAPES} chord classes)")
    except ImportError as e:
        print(f"ChordShapeCNN (P10) — skipped ({e})")

    print("\n✅ All models forward-pass OK.")