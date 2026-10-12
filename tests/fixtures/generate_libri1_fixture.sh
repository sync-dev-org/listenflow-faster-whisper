#!/usr/bin/env bash
set -euo pipefail

fixture_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache_dir="${LIBROSA_DATA_DIR:-${TMPDIR:-/tmp}/listenflow-librosa-cache}"
wav_path="${fixture_dir}/libri1_16khz_mono_s16le.wav"

# librosa "libri1": LibriSpeech SLR12, CC-BY 4.0.
source_ogg="$(
  LIBROSA_DATA_DIR="${cache_dir}" uv run \
    --isolated \
    --no-project \
    --with "librosa==0.11.0" \
    python - <<'PY'
import librosa

print(librosa.ex("libri1"))
PY
)"

ffmpeg -y -hide_banner -loglevel error \
  -i "${source_ogg}" \
  -ac 1 \
  -ar 16000 \
  -sample_fmt s16 \
  "${wav_path}"

file "${wav_path}"
