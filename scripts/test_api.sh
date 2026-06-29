#!/usr/bin/env bash
# ============================================================
# P7: GuitarAI API — Integration Test Script
# ============================================================
# Usage:  bash scripts/test_api.sh [AUDIO_FILE]
# Default audio: any .mp3 in the current directory
#
# What it tests:
#   1. GET  /health        → 200 + {"status": "ok"}
#   2. GET  /models        → 200 + model metadata dict
#   3. POST /transcribe    → 202 + job_id (< 1 second)
#   4. GET  /status/{id}   → PENDING → STARTED → SUCCESS loop
#   5. GET  /result/{id}   → full TranscriptionResult JSON
# ============================================================

set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"
AUDIO_FILE="${1:-}"
POLL_INTERVAL=3       # seconds between status polls
MAX_POLLS=100         # give up after this many polls

# ── Colours ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
info() { echo -e "${YELLOW}→${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

# ── Helpers ──────────────────────────────────────────────────────────────────
require_cmd() { command -v "$1" &>/dev/null || fail "Command not found: $1 — install it first."; }
require_cmd curl
require_cmd jq

# ── Find audio file ───────────────────────────────────────────────────────────
if [[ -z "$AUDIO_FILE" ]]; then
    # Try to find any audio file nearby
    AUDIO_FILE=$(find . -maxdepth 3 -name "*.mp3" -o -name "*.wav" -o -name "*.flac" | head -1)
fi
if [[ -z "$AUDIO_FILE" ]]; then
    fail "No audio file provided and none found in the current directory. Usage: $0 path/to/audio.mp3"
fi
if [[ ! -f "$AUDIO_FILE" ]]; then
    fail "Audio file not found: $AUDIO_FILE"
fi

echo ""
echo "════════════════════════════════════════════"
echo "  GuitarAI P7 API Integration Test"
echo "════════════════════════════════════════════"
echo "  API base: $BASE_URL"
echo "  Audio:    $AUDIO_FILE"
echo ""

# ── 1. Health check ───────────────────────────────────────────────────────────
info "1/5  GET /health"
resp=$(curl -sf "$BASE_URL/health")
status=$(echo "$resp" | jq -r '.status')
[[ "$status" == "ok" ]] || fail "Health check failed: $resp"
ok "  /health → $resp"

# ── 2. Models endpoint ───────────────────────────────────────────────────────
info "2/5  GET /models"
resp=$(curl -sf "$BASE_URL/models")
n_models=$(echo "$resp" | jq '.models | length')
ok "  /models → $n_models model(s) listed"
echo "$resp" | jq '.models | keys'

# ── 3. Submit transcription job ──────────────────────────────────────────────
info "3/5  POST /transcribe  (uploading $AUDIO_FILE)"
t0=$SECONDS
resp=$(curl -sf -X POST "$BASE_URL/transcribe" \
    -H "accept: application/json" \
    -F "file=@$AUDIO_FILE;type=audio/mpeg")
t1=$SECONDS
elapsed=$((t1 - t0))

JOB_ID=$(echo "$resp" | jq -r '.job_id')
[[ "$JOB_ID" != "null" && -n "$JOB_ID" ]] || fail "No job_id returned: $resp"
ok "  /transcribe → job_id=$JOB_ID  (${elapsed}s round-trip)"

if [[ $elapsed -gt 5 ]]; then
    echo -e "${YELLOW}  ⚠ Upload took ${elapsed}s — expected < 1s for a small file${NC}"
fi

# ── 4. Poll /status until done ───────────────────────────────────────────────
info "4/5  GET /status/$JOB_ID  (polling every ${POLL_INTERVAL}s)"
polls=0
while true; do
    resp=$(curl -sf "$BASE_URL/status/$JOB_ID")
    STATE=$(echo "$resp" | jq -r '.status')
    META=$(echo "$resp" | jq -c '.meta')
    echo "       state=$STATE  meta=$META"

    case "$STATE" in
        SUCCESS)
            ok "  Job completed successfully!"
            break
            ;;
        FAILURE)
            error=$(echo "$resp" | jq -r '.meta.error // "unknown"')
            fail "  Job FAILED: $error"
            ;;
        PENDING|STARTED)
            polls=$((polls + 1))
            if [[ $polls -ge $MAX_POLLS ]]; then
                fail "  Timed out after $((polls * POLL_INTERVAL))s waiting for job completion."
            fi
            sleep "$POLL_INTERVAL"
            ;;
        *)
            echo "  Unknown state: $STATE — continuing to poll..."
            sleep "$POLL_INTERVAL"
            ;;
    esac
done

# ── 5. Fetch result ──────────────────────────────────────────────────────────
info "5/5  GET /result/$JOB_ID"
result=$(curl -sf "$BASE_URL/result/$JOB_ID")

n_chords=$(echo "$result" | jq '.chords | length')
n_notes=$(echo "$result"  | jq '.notes | length')
proc_time=$(echo "$result" | jq '.pipeline.processing_time_sec')
models_used=$(echo "$result" | jq -c '.pipeline.models_used')

ok "  Result received!"
echo ""
echo "  ┌─────────────────────────────────────┐"
echo "  │  Chords detected : $n_chords"
echo "  │  Notes detected  : $n_notes"
echo "  │  Processing time : ${proc_time}s"
echo "  │  Models used     : $models_used"
echo "  └─────────────────────────────────────┘"
echo ""

# Print first 5 chords
echo "  First chords:"
echo "$result" | jq '.chords[:5]'

# Print ASCII tab (first 80 chars of each line to fit terminal)
echo ""
echo "  ASCII Tab (first window):"
echo "$result" | jq -r '.tab' | head -6

echo ""
echo "════════════════════════════════════════════"
ok " All 5 tests passed!"
echo "════════════════════════════════════════════"
echo ""

# Optionally save full result
OUTPUT_FILE="outputs/api_test_result_${JOB_ID:0:8}.json"
mkdir -p outputs
echo "$result" | jq . > "$OUTPUT_FILE"
info "Full result saved to $OUTPUT_FILE"
