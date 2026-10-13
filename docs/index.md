# listenflow-faster-whisper

`listenflow-faster-whisper` is an in-process faster-whisper engine for `listenflow`.
It converts 16 kHz mono PCM audio into asynchronous transcript events using a
caller-supplied offline model directory. Import its public API from
`listenflow.faster_whisper`.

The distribution depends on `listenflow>=0.17.0` and `faster-whisper==1.2.1`.
The caller supplies audio capture, the offline model files, and the startup
environment for the selected runtime profile.

## Install

Use Python 3.12 or 3.13:

```console
pip install listenflow-faster-whisper
```

### CUDA

The `cuda` extra declares Linux x86_64, including WSL2, as its host target. The
host supplies the NVIDIA driver, and the extra supplies the cuBLAS dependency:

```console
pip install 'listenflow-faster-whisper[cuda]'
CUDA_CUBLAS_LIBRARY_DIR="$(
  python -c \
    'from pathlib import Path; import nvidia.cublas.lib; print(Path(next(iter(nvidia.cublas.lib.__path__))))'
)"
export LD_LIBRARY_PATH="${CUDA_CUBLAS_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Set the library search path before starting Python. Request CUDA in the engine
constructor with `device="cuda"` and a compute type such as `"float16"`. The
engine leaves device compatibility to upstream and does not change the process
environment. Acceptance uses the repository's lock and caller-established startup
environment; other hosts and independently resolved environments remain unqualified.

## Use the engine

Provision a CTranslate2-format Whisper directory containing `config.json`,
`model.bin`, `tokenizer.json`, and either `vocabulary.json` or `vocabulary.txt`.
Include a matching `preprocessor_config.json` for models that require it, such as
`large-v3`. Supply an absolute directory path; the engine does not download weights.

```python
from collections.abc import AsyncIterator
from contextlib import aclosing

from listenflow import Final, OfflineModelDirectorySource
from listenflow.faster_whisper import FasterWhisperEngine


async def transcribe(
    model_directory: str,
    pcm_chunks: AsyncIterator[bytes],
) -> None:
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=model_directory),
        cpu_threads=2,
    )
    async with engine:
        async with aclosing(engine.stream(pcm_chunks)) as events:
            async for event in events:
                if isinstance(event, Final):
                    print(event.utterance_id, event.ts_offset_s, event.text)
```

Audio chunks must contain whole samples in one fixed format: signed 16-bit
little-endian PCM (`EngineInputFormat.PCM_S16LE`, the default) or little-endian
float32 PCM (`EngineInputFormat.PCM_F32`). Both use 16 kHz mono audio.

## Engine behavior

The constructor validates configuration types and ranges and reads the offline
metadata to validate language codes before loading weights. Language codes use
lower-case ISO 639-1 where available, otherwise ISO 639-3, without region or script
suffixes. The default is `en`; `None` requests language detection on multilingual
models and transcribes as English on English-only models. The code `jv` is passed
to faster-whisper as `jw`. Keep the directory unchanged until preparation.

`capabilities.languages` declares the immutable set of language codes accepted by
the configured model metadata. `capabilities.auto_language` declares whether
`language=None` uses detection, independently of the configured language: true for
a multilingual vocabulary and false for an English-only vocabulary. Invalid
settings and metadata raise `ConfigurationError` before recognition.

`prepare()` validates model assets and feature-extractor consistency, loads the
offline model, and drains a warm-up transcription using the full packaged synthetic
utterance (1.36 seconds). Warm-up uses the caller's decoding and silence-mitigation
settings and needs no network or
additional model assets. It produces no public events or health-counter progress.
Readiness and the immutable `runtime_status()` snapshot appear only after the load,
warm-up drain, and runtime readback all succeed. The snapshot distinguishes the
requested profile and model identity from effective runtime values and installed
runtime versions. `runtime_status()` raises `EngineStateError` before preparation
succeeds, including after a failed attempt.

Prepare explicitly before recognition, or use `async with engine` for preparation
on entry and closure on exit. Repeated successful preparation does nothing.
Missing native components and identifiable CUDA library or device failures raise
`EnvironmentUnavailableError` during preparation. The
[README preparation guide](https://github.com/sync-dev-org/listenflow-faster-whisper#preparation-and-closure)
describes measured loading time, including warm-up; no time limit is imposed by
the engine.

One stream can be active per engine instance. PCM intake continues while closed
utterances are decoded serially in close order, with event groups kept separate.
Normal end of input drains received utterances. Close the event iterator when
stopping early so that cleanup finishes before reuse. Upstream failures propagate
without profile substitution, retry, or fallback.

Caller cancellation, early iterator closure, and `await engine.aclose()` wait for
running native transcription to finish. Faster-whisper cannot stop it midway, so
cancellation has no fixed completion time. After iterator cleanup the prepared
engine can accept a new stream. Engine closure also releases the model, is
idempotent, and permanently closes the instance. Use one engine on one event loop.
An utterance yielding no text still emits its start and end events.

Errors use the core hierarchy exported from `listenflow`: `ConfigurationError`,
`EngineStateError`, `EngineClosedError`, `InferenceError`,
`EnvironmentUnavailableError`, and `ModuleNotInstalledError`, all under `STTError`.
Native and upstream errors are chained as their cause.

`health()` reports `state` (`unprepared`, `ready`, or `closed`), `busy`,
`last_progress_monotonic_s`, lifetime PCM and event counters, current decode and
pending utterances, and per-session prompt evidence. The progress timestamp advances
on processed PCM and completed utterance decoding, excluding warm-up.
Prompt evidence becomes applied after a successful decode drain and stays applied
for that session. `runtime_status()` is a fixed post-load snapshot; use `health()`
for live progress. Synchronous reads return held values without model work or lock
waiting. Health remains readable after engine closure, as defined by the core;
`capabilities` and `input_format` keep returning their values after closure.
`runtime_status()` raises `EngineClosedError` once closure starts, including while
native work is still finishing.

## Verification

The repository includes opt-in offline integration tests. Model and reference-audio
identities are pinned in `tests/test_faster_whisper_engine.py` and verified before
loading or decoding. CPU and CUDA checks use
`LISTENFLOW_FASTER_WHISPER_MODEL_DIR` for `small` and
`LISTENFLOW_FASTER_WHISPER_LARGE_V3_MODEL_DIR` for `large-v3`. CUDA checks also
require `LISTENFLOW_FASTER_WHISPER_CUDA_TEST=1` and the configured library search
path. Metadata-only checks use `LISTENFLOW_FASTER_WHISPER_SMALL_METADATA_DIR`,
`LISTENFLOW_FASTER_WHISPER_LARGE_V3_METADATA_DIR`, and
`LISTENFLOW_FASTER_WHISPER_SMALL_EN_METADATA_DIR`; they do not require `model.bin`.

The [repository README](https://github.com/sync-dev-org/listenflow-faster-whisper#opt-in-verification)
contains the verification commands and asset-directory setup.

## Versioning and license

Release versions come from exact `vX.Y.Z` tags and are published on PyPI. A
checkout without a matching tag uses the non-release fallback `0.0.0.dev0`.
The code and both packaged synthetic speech assets are licensed under MIT.
The model, runtime, voice, settings, spoken text, and conversion are recorded in
[the warm-up speech record](https://github.com/sync-dev-org/listenflow-faster-whisper/blob/main/src/listenflow/faster_whisper/warmup_speech_attribution.txt)
and
[the test-fixture speech record](https://github.com/sync-dev-org/listenflow-faster-whisper/blob/main/tests/fixtures/small_boat_speech_record.txt).

## API reference

See the [public engine and configuration API](api/faster_whisper.md) for constructor
parameters, methods, silence mitigation, and runtime-status fields.
