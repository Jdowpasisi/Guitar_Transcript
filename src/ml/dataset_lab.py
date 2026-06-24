"""
GuitarAI — Dataset Lab (P3)
=============================
Understand all three datasets. Build your data pipeline.
Set up train/val/test splits. Provide a unified get_sample() API.

Supported Datasets:
    1. GuitarSet (The Bedrock)  — JAMS annotations, 6 players, acoustic guitar
    2. Guitar-TECHS (The Electric Edge) — MIDI per-string, 3 players, electric
    3. IDMT-SMT-Guitar (The Diversity) — XML annotations, 7 guitars, techniques

Usage:
    python -m src.ml.dataset_lab              # Discover & summarize all datasets
    python -m src.ml.dataset_lab --verify     # Just verify dataset presence

Programmatic:
    from src.ml.dataset_lab import GuitarSetDataset, get_sample
    gs = GuitarSetDataset()
    gs.get_summary()
    splits = gs.create_splits()

    sample = get_sample("train", 0, dataset="guitarset")
    print(sample["chord_labels"])
    print(sample["tab_annotations"])
"""

import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Try to import shared config; fall back to defaults if run standalone
try:
    from src.config import (
        SR, DURATION_PREVIEW, DATASET_ROOTS,
        GUITARSET_AUDIO_MIC, GUITARSET_AUDIO_HEX, GUITARSET_ANNOTATIONS,
        GUITARSET_SPLIT_PLAYERS, GUITAR_TECHS_SPLIT_PLAYERS,
        DATA_SPLITS_DIR, OUTPUTS_DIR,
        CQT_HOP_LENGTH, CQT_FMIN, CQT_N_BINS, CQT_BINS_PER_OCT,
    )
except ImportError:
    # Fallback defaults for standalone usage
    SR = 22050
    DURATION_PREVIEW = 10
    DATASET_ROOTS = {
        "guitarset":    Path("data/raw/guitarset"),
        "guitar_techs": Path("data/raw/guitar_techs"),
        "idmt":         Path("data/raw/idmt_guitar"),
    }
    GUITARSET_AUDIO_MIC   = DATASET_ROOTS["guitarset"] / "audio_mono-mic"
    GUITARSET_AUDIO_HEX   = DATASET_ROOTS["guitarset"] / "audio_hex-pickup_debleeded"
    GUITARSET_ANNOTATIONS = DATASET_ROOTS["guitarset"] / "annotation"
    GUITARSET_SPLIT_PLAYERS = {
        "train": ["00", "01", "02", "03"],
        "val":   ["04"],
        "test":  ["05"],
    }
    GUITAR_TECHS_SPLIT_PLAYERS = {
        "train": ["P1"],
        "val":   ["P2"],
        "test":  ["P3"],
    }
    DATA_SPLITS_DIR = Path("data/splits")
    OUTPUTS_DIR     = Path("outputs")
    CQT_HOP_LENGTH = 512
    CQT_FMIN       = "E2"
    CQT_N_BINS     = 84
    CQT_BINS_PER_OCT = 12


# ═════════════════════════════════════════════════════════
#  JAMS PARSER (No `jams` library dependency)
# ═════════════════════════════════════════════════════════
# GuitarSet JAMS files are plain JSON. We parse them directly
# to avoid the `jams` library which has Python 3.14 issues.

def parse_jams(jams_path):
    """
    Parse a JAMS annotation file (plain JSON) and extract:
      - chord annotations  (namespace='chord' or 'chord_harte')
      - note MIDI events   (namespace='note_midi')
      - pitch contours      (namespace='pitch_contour')

    Returns
    -------
    dict with keys:
        chords : list of {time, duration, label}
        notes  : list of {time, duration, midi_pitch, string_index}
        pitch_contours : list of {time, duration, values, string_index}
        metadata : dict (file metadata from the JAMS sandbox)
    """
    with open(jams_path, "r") as f:
        data = json.load(f)

    result = {
        "chords": [],
        "notes": [],
        "pitch_contours": [],
        "metadata": data.get("file_metadata", {}),
    }

    # Track seen chords to avoid duplicates (GuitarSet has two chord
    # annotation blocks with the same data — one auto, one verified)
    seen_chords = set()

    for annotation in data.get("annotations", []):
        ns = annotation.get("namespace", "")
        ann_data = annotation.get("data", [])

        # --- Chord annotations ---
        if ns in ("chord", "chord_harte"):
            # data is a list of {time, duration, value, confidence} dicts
            if isinstance(ann_data, list):
                for obs in ann_data:
                    key = (obs["time"], obs["duration"], obs["value"])
                    if key not in seen_chords:
                        seen_chords.add(key)
                        result["chords"].append({
                            "time":     obs["time"],
                            "duration": obs["duration"],
                            "label":    obs["value"],
                        })

        # --- Note MIDI events ---
        # GuitarSet has 6 note_midi annotations (one per string, index 0-5)
        elif ns == "note_midi":
            string_idx = annotation.get("annotation_metadata", {}).get(
                "data_source", ""
            )
            # data_source in GuitarSet is "0", "1", ..., "5" (plain numbers)
            s_idx = _parse_string_index(string_idx, annotation)

            # data is a list of {time, duration, value, confidence} dicts
            if isinstance(ann_data, list):
                for obs in ann_data:
                    result["notes"].append({
                        "time":         obs["time"],
                        "duration":     obs["duration"],
                        "midi_pitch":   obs["value"],
                        "string_index": s_idx,
                    })

        # --- Pitch contours ---
        elif ns == "pitch_contour":
            string_idx = annotation.get("annotation_metadata", {}).get(
                "data_source", ""
            )
            s_idx = _parse_string_index(string_idx, annotation)

            # GuitarSet pitch_contour data is a dict of parallel arrays:
            #   {"time": [...], "duration": [...], "value": [...], ...}
            # NOT a list of dicts like note_midi
            if isinstance(ann_data, dict):
                times = ann_data.get("time", [])
                durations = ann_data.get("duration", [])
                values = ann_data.get("value", [])
                # Store as a single contour entry per string
                result["pitch_contours"].append({
                    "times":        times,
                    "durations":    durations,
                    "values":       values,
                    "string_index": s_idx,
                    "num_points":   len(times) if isinstance(times, list) else 0,
                })
            elif isinstance(ann_data, list):
                for obs in ann_data:
                    result["pitch_contours"].append({
                        "time":         obs.get("time"),
                        "duration":     obs.get("duration"),
                        "values":       obs.get("value"),
                        "string_index": s_idx,
                    })

    return result


def _parse_string_index(data_source, annotation):
    """Extract string index (0-5) from JAMS annotation metadata.

    GuitarSet uses plain numeric strings as data_source:
        "0", "1", "2", "3", "4", "5"  → string indices
    """
    # GuitarSet: data_source is a plain number string like "0", "1", ..., "5"
    if isinstance(data_source, (int, float)):
        return int(data_source)
    if isinstance(data_source, str):
        # Try plain number first ("0", "1", etc.)
        stripped = data_source.strip()
        try:
            idx = int(stripped)
            if 0 <= idx <= 5:
                return idx
        except ValueError:
            pass
        # Try "stringN" format as fallback
        if "string" in stripped.lower():
            try:
                return int(stripped.lower().replace("string", "").strip())
            except ValueError:
                pass

    # Fallback: try the annotation_metadata's index field
    idx = annotation.get("annotation_metadata", {}).get("index", None)
    if idx is not None:
        return int(idx)

    return -1  # Unknown string


# ═════════════════════════════════════════════════════════
#  XML PARSER (IDMT-SMT-Guitar)
# ═════════════════════════════════════════════════════════

def parse_idmt_xml(xml_path):
    """
    Parse an IDMT-SMT-Guitar XML annotation file.
    
    Returns
    -------
    list of dict
        Each dict: {onset, offset, midi_pitch, technique, fret, string}
        Fields may be None if not present in the XML.
    """
    events = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # IDMT XML structure: <instrumentRecording> > <transcription> > <event>
        # Each <event> has: pitch, onsetSec, offsetSec, fretNumber,
        #   stringNumber, excitationStyle (plucking), expressionStyle (technique)
        for event in root.iter("event"):
            excitation = _xml_text(event, "excitationStyle")   # PK, FS, MU
            expression = _xml_text(event, "expressionStyle")   # NO, BE, SL, VI, HA, DN
            e = {
                "onset":            _xml_float(event, "onsetSec"),
                "offset":           _xml_float(event, "offsetSec"),
                "midi_pitch":       _xml_float(event, "pitch"),
                "excitation_style": excitation,
                "expression_style": expression,
                # Combine for a single "technique" label (backwards compatible)
                "technique":        expression if expression and expression != "NO" else excitation,
                "fret":             _xml_int(event, "fretNumber"),
                "string":           _xml_int(event, "stringNumber"),
            }
            # Fallback field names used in some subsets
            if e["onset"] is None:
                e["onset"] = _xml_float(event, "onset")
            if e["offset"] is None:
                e["offset"] = _xml_float(event, "offset")
            events.append(e)

        # Some subsets use <note> elements instead of <event>
        if not events:
            for note in root.iter("note"):
                events.append({
                    "onset":            _xml_float(note, "onsetSec") or _xml_float(note, "onset"),
                    "offset":           _xml_float(note, "offsetSec") or _xml_float(note, "offset"),
                    "midi_pitch":       _xml_float(note, "pitch") or _xml_float(note, "midiPitch"),
                    "excitation_style": _xml_text(note, "excitationStyle"),
                    "expression_style": _xml_text(note, "expressionStyle"),
                    "technique":        _xml_text(note, "excitationStyle") or _xml_text(note, "technique"),
                    "fret":             _xml_int(note, "fretNumber") or _xml_int(note, "fret"),
                    "string":           _xml_int(note, "stringNumber") or _xml_int(note, "string"),
                })
    except ET.ParseError as e:
        print(f"  ⚠️  XML parse error in {xml_path}: {e}")

    return events


def _xml_float(elem, tag):
    """Safely extract a float from an XML element's child."""
    child = elem.find(tag)
    if child is not None and child.text:
        try:
            return float(child.text.strip())
        except ValueError:
            pass
    return None


def _xml_int(elem, tag):
    """Safely extract an int from an XML element's child."""
    child = elem.find(tag)
    if child is not None and child.text:
        try:
            return int(float(child.text.strip()))
        except ValueError:
            pass
    return None


def _xml_text(elem, tag):
    """Safely extract text from an XML element's child."""
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None


# ═════════════════════════════════════════════════════════
#  DATASET: GuitarSet (The Bedrock)
# ═════════════════════════════════════════════════════════
class GuitarSetDataset:
    """
    GuitarSet (ISMIR 2018): 360 annotated guitar recordings from 6 players.
    Annotations include chords, note MIDI, pitch contours, and (string, fret).
    This is the primary training data for P4 (Chord CNN), P5 (Benchmarker),
    and P6 (Voicing LSTM).

    Parameters
    ----------
    root_dir : str or Path, optional
        Root directory of GuitarSet. Defaults to config value.
    audio_source : str
        Which audio to use: 'mic' (mono-mic) or 'hex' (hex-pickup debleeded).
    """
    def __init__(self, root_dir=None, audio_source="mic"):
        self.root_dir = Path(root_dir) if root_dir else DATASET_ROOTS["guitarset"]
        self.audio_source = audio_source

        if audio_source == "hex":
            self.audio_dir = self.root_dir / "audio_hex-pickup_debleeded"
        else:
            self.audio_dir = self.root_dir / "audio_mono-mic"

        self.annotation_dir = self.root_dir / "annotation"

        # Discover files
        self.audio_files = sorted(self.audio_dir.rglob("*.wav")) if self.audio_dir.exists() else []
        self.jams_files  = sorted(self.annotation_dir.rglob("*.jams")) if self.annotation_dir.exists() else []

        # Build a mapping: stem → {audio_path, jams_path}
        self._pairs = self._build_pairs()

    def _build_pairs(self):
        """
        Match audio files to their JAMS annotations by stem name.
        GuitarSet naming: "00_BN1-129-Eb_comp_hex_cln.wav" (audio)
                          "00_BN1-129-Eb_comp.jams" (annotation)
        The JAMS stem is a prefix of the audio stem.
        """
        pairs = {}
        jams_by_stem = {}
        for jp in self.jams_files:
            jams_by_stem[jp.stem] = jp

        for ap in self.audio_files:
            audio_stem = ap.stem
            # Try exact match first, then prefix matching
            matched_jams = None
            if audio_stem in jams_by_stem:
                matched_jams = jams_by_stem[audio_stem]
            else:
                # GuitarSet audio stems have suffixes like _hex_cln, _mic
                # JAMS stems are the base: "00_BN1-129-Eb_comp"
                for js, jp in jams_by_stem.items():
                    if audio_stem.startswith(js):
                        matched_jams = jp
                        break

            pairs[audio_stem] = {
                "audio_path": ap,
                "jams_path":  matched_jams,
            }

        return pairs

    def get_player_id(self, filename):
        """Extract player ID (first 2 chars) from a GuitarSet filename."""
        return Path(filename).stem[:2]

    def get_summary(self):
        """Print a summary of the GuitarSet dataset."""
        print("─── 📊 GuitarSet Summary ───")
        print(f"  Audio source  : {self.audio_source} ({self.audio_dir.name})")
        print(f"  Audio files   : {len(self.audio_files)}")
        print(f"  JAMS files    : {len(self.jams_files)}")
        print(f"  Matched pairs : {sum(1 for p in self._pairs.values() if p['jams_path'])}")

        # Player distribution
        player_counts = {}
        for stem, pair in self._pairs.items():
            pid = self.get_player_id(stem)
            player_counts[pid] = player_counts.get(pid, 0) + 1
        print(f"  Players       : {len(player_counts)}")
        for pid in sorted(player_counts):
            print(f"    Player {pid}: {player_counts[pid]} recordings")

        # Chord vocabulary (sample from first few JAMS files)
        if self.jams_files:
            chord_set = set()
            for jp in self.jams_files[:10]:
                parsed = parse_jams(jp)
                for c in parsed["chords"]:
                    chord_set.add(c["label"])
            print(f"  Chord vocab (sample of 10 files): {len(chord_set)} labels")
            if chord_set:
                sample = sorted(chord_set)[:15]
                print(f"    e.g.: {', '.join(sample)}")

    def create_splits(self, out_path=None, player_split=None):
        """
        Split by player to prevent data leakage.
        Different players = different timbre/style → proper generalization test.

        Parameters
        ----------
        out_path : str or Path, optional
            Where to save the splits JSON. Default: data/splits/splits_guitarset.json
        player_split : dict, optional
            Mapping of split name → list of player IDs.
            Default: GUITARSET_SPLIT_PLAYERS from config.

        Returns
        -------
        dict : {train: [paths], val: [paths], test: [paths]}
        """
        if out_path is None:
            out_path = DATA_SPLITS_DIR / "splits_guitarset.json"
        if player_split is None:
            player_split = GUITARSET_SPLIT_PLAYERS

        splits = {"train": [], "val": [], "test": []}

        for stem, pair in self._pairs.items():
            pid = self.get_player_id(stem)
            for split_name, players in player_split.items():
                if pid in players:
                    splits[split_name].append({
                        "audio": str(pair["audio_path"]),
                        "jams":  str(pair["jams_path"]) if pair["jams_path"] else None,
                        "player": pid,
                    })
                    break

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"✅ GuitarSet splits saved → {out_path}")
        for name, items in splits.items():
            players_in_split = set(item["player"] for item in items)
            print(f"   {name:5s}: {len(items):3d} recordings "
                  f"(players: {', '.join(sorted(players_in_split))})")

        return splits

    def get_annotations(self, audio_path):
        """
        Get parsed annotations for an audio file.

        Returns
        -------
        dict or None
            Parsed JAMS data with chords, notes, pitch_contours, metadata.
        """
        audio_stem = Path(audio_path).stem
        pair = self._pairs.get(audio_stem)
        if pair and pair["jams_path"]:
            return parse_jams(pair["jams_path"])
        return None


# ═════════════════════════════════════════════════════════
#  DATASET: Guitar-TECHS (The Electric Edge)
# ═════════════════════════════════════════════════════════
class GuitarTECHSDataset:
    """
    Guitar-TECHS (ICASSP 2025): Electric guitar dataset covering
    techniques, musical excerpts, chords, and scales across 3 players
    with diverse hardware.

    Parameters
    ----------
    root_dir : str or Path, optional
        Root directory. Defaults to config value.
    """
    def __init__(self, root_dir=None):
        self.root_dir = Path(root_dir) if root_dir else DATASET_ROOTS["guitar_techs"]
        self.audio_files = sorted(self.root_dir.rglob("*.wav")) if self.root_dir.exists() else []
        self.midi_files  = sorted(self.root_dir.rglob("*.mid")) if self.root_dir.exists() else []

    def get_player_id(self, filepath):
        """
        Extract player ID from Guitar-TECHS file path.
        Files are in directories like P1_chords/, P2_techniques/, P3_music/.
        """
        filepath = Path(filepath)
        for part in filepath.parts:
            for pid in ["P1", "P2", "P3"]:
                if part.startswith(pid):
                    return pid
        return "unknown"

    def get_category(self, filepath):
        """
        Extract category (chords, scales, techniques, music, singlenotes)
        from directory name.
        """
        filepath = Path(filepath)
        for part in filepath.parts:
            part_lower = part.lower()
            for cat in ["chords", "scales", "techniques", "music",
                        "singlenotes"]:
                if cat in part_lower:
                    return cat
        return "unknown"

    def get_summary(self):
        """Print a summary of the Guitar-TECHS dataset."""
        print("─── 📊 Guitar-TECHS Summary ───")
        print(f"  Total WAV files : {len(self.audio_files)}")
        print(f"  Total MIDI files: {len(self.midi_files)}")

        # Player distribution
        player_counts = {}
        for f in self.audio_files:
            pid = self.get_player_id(f)
            player_counts[pid] = player_counts.get(pid, 0) + 1
        for pid in sorted(player_counts):
            print(f"    {pid}: {player_counts[pid]} files")

        # Category distribution
        cat_counts = {}
        for f in self.audio_files:
            cat = self.get_category(f)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"  Categories:")
        for cat in sorted(cat_counts):
            print(f"    {cat:15s}: {cat_counts[cat]} files")

    def create_splits(self, out_path=None, player_split=None, seed=42):
        """
        Split by player to ensure timbre generalization.
        Guitar-TECHS has 3 players: P1, P2, P3.

        Parameters
        ----------
        out_path : str or Path, optional
            Default: data/splits/splits_guitar_techs.json
        player_split : dict, optional
            Default: GUITAR_TECHS_SPLIT_PLAYERS from config.

        Returns
        -------
        dict : {train: [paths], val: [paths], test: [paths]}
        """
        if out_path is None:
            out_path = DATA_SPLITS_DIR / "splits_guitar_techs.json"
        if player_split is None:
            player_split = GUITAR_TECHS_SPLIT_PLAYERS

        splits = {"train": [], "val": [], "test": []}

        for f in self.audio_files:
            pid = self.get_player_id(f)
            entry = {
                "audio":    str(f),
                "player":   pid,
                "category": self.get_category(f),
            }
            assigned = False
            for split_name, players in player_split.items():
                if pid in players:
                    splits[split_name].append(entry)
                    assigned = True
                    break
            if not assigned:
                # Files from unknown players go to train
                splits["train"].append(entry)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"✅ Guitar-TECHS splits saved → {out_path}")
        for name, items in splits.items():
            players_in_split = set(item["player"] for item in items)
            print(f"   {name:5s}: {len(items):3d} files "
                  f"(players: {', '.join(sorted(players_in_split))})")

        return splits


# ═════════════════════════════════════════════════════════
#  DATASET: IDMT-SMT-Guitar (The Diversity)
# ═════════════════════════════════════════════════════════
class IDMTGuitarDataset:
    """
    IDMT-SMT-Guitar (Fraunhofer): 7 guitars, multiple techniques,
    44.1kHz/24-bit WAV with XML annotations.
    Best for technique classification (bend, slide, vibrato, harmonics).

    Parameters
    ----------
    root_dir : str or Path, optional
        Root directory. Defaults to config value.
    """
    def __init__(self, root_dir=None):
        self.root_dir = Path(root_dir) if root_dir else DATASET_ROOTS["idmt"]
        self.audio_files = sorted(self.root_dir.rglob("*.wav")) if self.root_dir.exists() else []
        self.xml_files   = sorted(self.root_dir.rglob("*.xml")) if self.root_dir.exists() else []

        # Build audio → XML mapping
        self._annotation_map = self._build_annotation_map()

    def _build_annotation_map(self):
        """Map audio file stems to their XML annotation paths."""
        xml_by_stem = {}
        for xp in self.xml_files:
            xml_by_stem[xp.stem] = xp

        mapping = {}
        for ap in self.audio_files:
            # Try exact stem match
            if ap.stem in xml_by_stem:
                mapping[str(ap)] = xml_by_stem[ap.stem]
            else:
                # Some IDMT files have annotation XMLs with slightly different names
                # Try prefix matching
                for xs, xp in xml_by_stem.items():
                    if ap.stem.startswith(xs) or xs.startswith(ap.stem):
                        mapping[str(ap)] = xp
                        break
        return mapping

    def get_subset(self, filepath):
        """Identify which IDMT subset (1-4) a file belongs to."""
        filepath = Path(filepath)
        try:
            rel = filepath.relative_to(self.root_dir)
        except ValueError:
            rel = filepath
        rel_str = str(rel).lower()
        for i in range(1, 5):
            if f"dataset{i}" in rel_str or f"subset{i}" in rel_str:
                return i
        return 0  # Unknown

    def get_guitar_id(self, filepath):
        """Extract guitar/instrument identifier from path structure.

        IDMT actual layout:
            idmt_guitar/IDMT-SMT-GUITAR_V2/dataset1/Guitar Name/audio/file.wav
            idmt_guitar/IDMT-SMT-GUITAR_V2/dataset4/Guitar Name/fast/genre/file.wav

        We combine dataset + guitar name as the grouping key to get
        meaningful instrument-level splits.
        """
        filepath = Path(filepath)
        try:
            rel = filepath.relative_to(self.root_dir)
            parts = rel.parts  # e.g. ('IDMT-SMT-GUITAR_V2', 'dataset1', 'Fender...', 'audio', 'file.wav')
            # Skip 'IDMT-SMT-GUITAR_V2' wrapper, use dataset+guitar
            if len(parts) >= 3 and parts[0].startswith("IDMT"):
                return f"{parts[1]}/{parts[2]}"  # e.g. "dataset1/Fender Strat Clean Neck SC"
            elif len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"  # fallback
            elif len(parts) >= 1:
                return str(parts[0])
            else:
                return "unknown"
        except ValueError:
            return "unknown"

    def get_summary(self):
        """Print a summary of the IDMT-SMT-Guitar dataset."""
        print("─── 📊 IDMT-SMT-Guitar Summary ───")
        print(f"  Total WAV files : {len(self.audio_files)}")
        print(f"  Total XML annots: {len(self.xml_files)}")
        print(f"  Annotated audio : {len(self._annotation_map)}")

        # Guitar/subset distribution
        guitar_groups = {}
        for f in self.audio_files:
            key = self.get_guitar_id(f)
            guitar_groups.setdefault(key, []).append(f)
        print(f"  Guitar groups   : {len(guitar_groups)}")
        for gid in sorted(guitar_groups):
            print(f"    {gid:20s}: {len(guitar_groups[gid])} files")

        # Technique distribution (from parsed XMLs)
        if self._annotation_map:
            technique_counts = {}
            sample_count = min(50, len(self._annotation_map))
            for i, (audio_path, xml_path) in enumerate(self._annotation_map.items()):
                if i >= sample_count:
                    break
                events = parse_idmt_xml(xml_path)
                for e in events:
                    t = e.get("technique") or "unspecified"
                    technique_counts[t] = technique_counts.get(t, 0) + 1
            if technique_counts:
                print(f"  Techniques (sample of {sample_count} files):")
                for t in sorted(technique_counts):
                    print(f"    {t:20s}: {technique_counts[t]} events")

    def get_annotations(self, audio_path):
        """
        Get parsed XML annotations for an audio file.

        Returns
        -------
        list of dict or None
            Each dict: {onset, offset, midi_pitch, technique, fret, string}
        """
        xml_path = self._annotation_map.get(str(audio_path))
        if xml_path:
            return parse_idmt_xml(xml_path)
        return None

    def create_splits(self, out_path=None, val_frac=0.1, test_frac=0.2,
                      seed=42):
        """
        Split by guitar folder (different instrument = different timbre).
        This prevents timbre leakage between train/test — same principle
        as the player-wise split in GuitarSet.

        Returns
        -------
        dict : {train: [entries], val: [entries], test: [entries]}
        """
        if out_path is None:
            out_path = DATA_SPLITS_DIR / "splits_idmt.json"

        np.random.seed(seed)

        # Group by guitar (top-level subdirectory)
        guitar_groups = {}
        for f in self.audio_files:
            key = self.get_guitar_id(f)
            guitar_groups.setdefault(key, []).append(str(f))

        guitars = sorted(guitar_groups.keys())
        np.random.shuffle(guitars)

        n = len(guitars)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))

        def make_entries(keys):
            entries = []
            for k in keys:
                for fpath in guitar_groups[k]:
                    has_annot = str(fpath) in self._annotation_map
                    entries.append({
                        "audio":      fpath,
                        "guitar_id":  k,
                        "has_annotation": has_annot,
                    })
            return entries

        splits = {
            "train": make_entries(guitars[n_test + n_val:]),
            "val":   make_entries(guitars[n_test:n_test + n_val]),
            "test":  make_entries(guitars[:n_test]),
        }

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"✅ IDMT splits saved → {out_path}")
        for name, items in splits.items():
            guitars_in_split = set(item["guitar_id"] for item in items)
            print(f"   {name:5s}: {len(items):3d} files "
                  f"(guitars: {', '.join(sorted(guitars_in_split))})")

        return splits


# ═════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (Pro-Level Audio Cleaning Baked In)
# ═════════════════════════════════════════════════════════
def extract_cqt(audio_path, sr=SR, duration=None):
    """
    CQT is preferred over mel-spectrogram for guitar because it has
    finer frequency resolution at low frequencies, and pitch shifts
    become simple bin translations — critical for chord recognition.

    Pro audio cleaning applied here:
      1. High-pass filter at 80Hz  (removes mud below lowest guitar note)
      2. LUFS-style peak normalization

    Parameters
    ----------
    audio_path : str or Path
    sr : int
    duration : float, optional

    Returns
    -------
    np.ndarray
        CQT magnitude, shape (n_bins, time_frames)
    """
    y, _ = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)

    # 1. High-pass filter (HPF) — removes sub-80Hz rumble
    y = librosa.effects.preemphasis(y)  # soft HPF; tune cutoff with scipy if needed

    # 2. Peak normalization (prevents quiet/loud confusion)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # 3. CQT with guitar-optimized settings
    C = librosa.cqt(
        y,
        sr=sr,
        hop_length=CQT_HOP_LENGTH,
        fmin=librosa.note_to_hz(CQT_FMIN),
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCT,
    )
    return np.abs(C)


# ═════════════════════════════════════════════════════════
#  VISUALIZATION
# ═════════════════════════════════════════════════════════
def visualize_sample(audio_path, index=0, out_dir=None):
    """Visualize a single audio file: CQT spectrogram + waveform."""
    if out_dir is None:
        out_dir = OUTPUTS_DIR / "viz"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    C = extract_cqt(audio_path, duration=DURATION_PREVIEW)
    C_db = librosa.amplitude_to_db(C, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: CQT
    img = librosa.display.specshow(
        C_db, sr=SR, hop_length=CQT_HOP_LENGTH,
        x_axis="time", y_axis="cqt_note",
        fmin=librosa.note_to_hz(CQT_FMIN),
        ax=axes[0]
    )
    axes[0].set_title(f"CQT Spectrogram — {Path(audio_path).name}")
    fig.colorbar(img, ax=axes[0], format="%+2.0f dB")

    # Bottom: waveform
    y, _ = librosa.load(str(audio_path), sr=SR, duration=DURATION_PREVIEW)
    librosa.display.waveshow(y, sr=SR, ax=axes[1])
    axes[1].set_title("Waveform")

    plt.tight_layout()
    out_path = out_dir / f"sample_viz_{index}.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def visualize_n_samples(file_list, n=5, out_dir=None):
    """Visualize N samples from a file list."""
    print(f"\n📸 Visualizing {min(n, len(file_list))} samples...")
    for i, item in enumerate(file_list[:n]):
        # Handle both plain paths and dict entries from splits
        path = item["audio"] if isinstance(item, dict) else item
        visualize_sample(path, index=i, out_dir=out_dir)


# ═════════════════════════════════════════════════════════
#  UNIFIED get_sample API
# ═════════════════════════════════════════════════════════
def get_sample(split_name, index, dataset="guitarset",
               splits_dir=None):
    """
    Get a single sample with audio features AND labels.

    This is the drop-in API that P4 (Chord CNN), P5 (Benchmarker),
    and P6 (Voicing LSTM) all call.

    Parameters
    ----------
    split_name : str
        'train', 'val', or 'test'
    index : int
        Index within the split
    dataset : str
        'guitarset', 'guitar_techs', or 'idmt'
    splits_dir : str or Path, optional
        Directory containing split JSON files

    Returns
    -------
    dict
        Keys: file, audio, cqt, shape, chord_labels, tab_annotations, metadata
    """
    if splits_dir is None:
        splits_dir = DATA_SPLITS_DIR

    splits_file = Path(splits_dir) / f"splits_{dataset}.json"
    if not splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}\n"
            f"Run the dataset's create_splits() method first."
        )

    with open(splits_file) as f:
        splits = json.load(f)

    if split_name not in splits:
        raise ValueError(f"Unknown split '{split_name}'. "
                         f"Available: {list(splits.keys())}")

    if index >= len(splits[split_name]):
        raise IndexError(f"Index {index} out of range for '{split_name}' "
                         f"split (size: {len(splits[split_name])})")

    entry = splits[split_name][index]

    # Entry can be a dict (new format) or a plain string (legacy)
    if isinstance(entry, dict):
        file_path = entry["audio"]
    else:
        file_path = entry

    # Load audio and extract CQT
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    cqt = extract_cqt(file_path)

    # Parse annotations based on dataset type
    chord_labels = []
    tab_annotations = []
    metadata = {}

    if dataset == "guitarset":
        jams_path = entry.get("jams") if isinstance(entry, dict) else None
        if jams_path and Path(jams_path).exists():
            parsed = parse_jams(jams_path)
            chord_labels = parsed["chords"]
            metadata = parsed["metadata"]

            # Convert note events to tab annotations
            for note in parsed["notes"]:
                tab_annotations.append({
                    "time":         note["time"],
                    "duration":     note["duration"],
                    "midi_pitch":   note["midi_pitch"],
                    "string_index": note["string_index"],
                    # Fret is computed from MIDI pitch and string tuning
                    "fret":         _midi_to_fret(note["midi_pitch"],
                                                  note["string_index"]),
                })

    elif dataset == "idmt":
        audio_path_str = str(file_path)
        # Rebuild annotation map for lookup
        idmt = IDMTGuitarDataset()
        events = idmt.get_annotations(audio_path_str)
        if events:
            for e in events:
                tab_annotations.append({
                    "time":       e["onset"],
                    "duration":   (e["offset"] - e["onset"]) if e["offset"] and e["onset"] else None,
                    "midi_pitch": e["midi_pitch"],
                    "string":     e["string"],
                    "fret":       e["fret"],
                    "technique":  e["technique"],
                })

    elif dataset == "guitar_techs":
        if isinstance(entry, dict):
            metadata = {
                "player":   entry.get("player"),
                "category": entry.get("category"),
            }

    return {
        "file":             file_path,
        "audio":            audio,
        "cqt":              cqt,          # shape: (84, T)
        "shape":            cqt.shape,
        "chord_labels":     chord_labels,
        "tab_annotations":  tab_annotations,
        "metadata":         metadata,
    }


# Standard guitar tuning — GuitarSet convention:
#   string 0 = lowest (E2), string 5 = highest (E4)
# MIDI note numbers for open strings
_GUITAR_OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4


def _midi_to_fret(midi_pitch, string_index):
    """
    Convert a MIDI pitch to a fret number given the string index.
    Returns None if the string index is invalid or the fret would be negative.
    """
    if string_index < 0 or string_index >= len(_GUITAR_OPEN_STRINGS):
        return None
    fret = int(round(midi_pitch)) - _GUITAR_OPEN_STRINGS[string_index]
    if fret < 0:
        return None
    return fret


# ═════════════════════════════════════════════════════════
#  UNIFIED DATASET WRAPPER
# ═════════════════════════════════════════════════════════
class UnifiedDataset:
    """
    Wraps all three datasets behind a common interface.
    Useful for training models on mixed data from multiple sources.

    Usage:
        unified = UnifiedDataset()
        unified.get_summary()
        unified.create_all_splits()
    """
    def __init__(self):
        self.guitarset    = GuitarSetDataset()
        self.guitar_techs = GuitarTECHSDataset()
        self.idmt         = IDMTGuitarDataset()

    def get_summary(self):
        """Print summaries for all available datasets."""
        if self.guitarset.audio_files:
            self.guitarset.get_summary()
        else:
            print("─── ⚠️  GuitarSet: NOT FOUND ───")

        print()

        if self.guitar_techs.audio_files:
            self.guitar_techs.get_summary()
        else:
            print("─── ⚠️  Guitar-TECHS: NOT FOUND ───")

        print()

        if self.idmt.audio_files:
            self.idmt.get_summary()
        else:
            print("─── ⚠️  IDMT-SMT-Guitar: NOT FOUND ───")

    def create_all_splits(self):
        """Create train/val/test splits for all available datasets."""
        results = {}
        if self.guitarset.audio_files:
            results["guitarset"] = self.guitarset.create_splits()
        if self.guitar_techs.audio_files:
            results["guitar_techs"] = self.guitar_techs.create_splits()
        if self.idmt.audio_files:
            results["idmt"] = self.idmt.create_splits()
        return results


# ═════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="GuitarAI Dataset Lab — Discover & prepare all datasets")
    parser.add_argument("--verify", action="store_true",
                        help="Just verify dataset presence, don't create splits")
    args = parser.parse_args()

    if args.verify:
        try:
            from src.config import verify_datasets
            verify_datasets()
        except ImportError:
            print("Run from project root: python -m src.ml.dataset_lab --verify")
        exit()

    # Full discovery and split creation
    print("=" * 60)
    print("  🗃  GuitarAI Dataset Lab")
    print("=" * 60)

    unified = UnifiedDataset()
    unified.get_summary()

    print("\n" + "=" * 60)
    print("  📊 Creating Splits")
    print("=" * 60)
    splits = unified.create_all_splits()

    # Test get_sample() if GuitarSet splits were created
    if "guitarset" in splits and splits["guitarset"]["train"]:
        print("\n" + "=" * 60)
        print("  🧪 get_sample() test (GuitarSet)")
        print("=" * 60)
        try:
            sample = get_sample("train", 0, dataset="guitarset")
            print(f"  File   : {sample['file']}")
            print(f"  Audio  : shape={sample['audio'].shape}, "
                  f"duration={len(sample['audio'])/SR:.2f}s")
            print(f"  CQT    : {sample['shape']}  (freq_bins × time_frames)")
            print(f"  Chords : {len(sample['chord_labels'])} labels")
            if sample["chord_labels"]:
                first = sample["chord_labels"][0]
                print(f"    First: {first['label']} @ {first['time']:.2f}s")
            print(f"  Tab    : {len(sample['tab_annotations'])} note events")
            if sample["tab_annotations"]:
                first = sample["tab_annotations"][0]
                print(f"    First: MIDI {first['midi_pitch']} → "
                      f"string {first['string_index']}, fret {first['fret']}")
        except Exception as e:
            print(f"  ⚠️  get_sample() failed: {e}")
            print("  This is expected if the splits JSON contains paths "
                  "that haven't been downloaded yet.")

    # Visualize a few samples from each dataset
    for ds_name, ds_splits in splits.items():
        if ds_splits["train"]:
            print(f"\n📸 Visualizing {ds_name} samples...")
            out = OUTPUTS_DIR / "viz" / ds_name
            visualize_n_samples(ds_splits["train"], n=3,
                                out_dir=str(out))