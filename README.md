# listenflow-faster-whisper

`listenflow-faster-whisper` provides an in-process faster-whisper engine for
`listenflow`. It turns a stream of 16 kHz mono PCM audio into asynchronous
transcript events using a caller-supplied offline model directory. The distribution
name is `listenflow-faster-whisper`; its Python import name is
`listenflow.faster_whisper`.

The wheel declares `listenflow>=0.17.0` as a normal dependency on the core
library. It also declares `faster-whisper==1.2.1` as its engine dependency.

## Install

Requires Python 3.12 or 3.13.

```console
pip install listenflow-faster-whisper
```

### CUDA

The declared host target for the `cuda` extra is Linux x86_64, including WSL2. The
host supplies the NVIDIA driver; the extra supplies the cuBLAS dependency. Install
the extra in your Python environment and set the library search path before
starting the process that uses the engine:

```console
pip install 'listenflow-faster-whisper[cuda]'
CUDA_CUBLAS_LIBRARY_DIR="$(
  python -c \
    'from pathlib import Path; import nvidia.cublas.lib; print(Path(next(iter(nvidia.cublas.lib.__path__))))'
)"
export LD_LIBRARY_PATH="${CUDA_CUBLAS_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

The caller owns `LD_LIBRARY_PATH`; the package does not rewrite the process
environment or select a device. To request GPU inference, pass `device="cuda"` and
the desired `compute_type`, such as `"float16"`, to the constructor. Acceptance is
verified with the repository's lock and the caller-established startup environment.
Other host configurations and independently resolved consumer environments remain
unqualified.

## Use the engine

Provision a CTranslate2-format Whisper model directory before constructing the
engine. It must contain `config.json`, `model.bin`, `tokenizer.json`, and either
`vocabulary.json` or `vocabulary.txt`. Include a matching
`preprocessor_config.json` when the model needs it, such as `large-v3`. The engine
does not discover, download, or cache weights.

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

The caller supplies an absolute directory path and audio chunks. The default input
format is signed 16-bit little-endian PCM (`EngineInputFormat.PCM_S16LE`);
little-endian float32 PCM (`EngineInputFormat.PCM_F32`) is also supported through the
`input_format` constructor argument. Both represent 16 kHz mono audio, and each
chunk must contain whole samples. Keep the input format fixed for the stream.

## Engine contract

### Profiles and language

The constructor accepts model labels, device and compute-type strings, CPU thread
count, beam size, decoding limits, prompt, word-timestamp settings, silence
mitigation, and utterance-boundary settings. It validates types and ranges and
leaves runtime compatibility to upstream. The `model` label does not select or
verify weights; `model_source` is their sole source. The engine does not substitute
profiles, retry inference, or fall back from CUDA to CPU.

At construction, `FasterWhisperEngine` reads the caller's offline model metadata
without loading weights and accepts only language codes that model handles. An
English-only vocabulary accepts `en`; a multilingual vocabulary uses its listed
`lang_ids` tokens that faster-whisper can use. Text vocabularies follow
CTranslate2's line-ending rule when checking for the empty-string token. Codes use
lower-case ISO 639-1
when available, otherwise ISO 639-3, without a region or script suffix. The
contract code `jv` is passed to faster-whisper as `jw`. Invalid values raise at
construction with `ConfigurationError`, before model load; the error suggests a code only when the model
handles that language and otherwise lists its accepted codes. The caller keeps
the directory unchanged until `prepare()`. The default language remains `en`;
`language=None` lets faster-whisper detect a language on multilingual models and
transcribes as `en` on English-only models. `capabilities.languages` is the
immutable set of accepted codes read from that metadata. `capabilities.auto_language`
is true for a multilingual vocabulary and false for an English-only vocabulary,
independently of the configured `language`. Missing, unreadable, or malformed
metadata and invalid configuration values raise `ConfigurationError` with the
original validation or file error as their cause.

### Preparation and closure

Call `await engine.prepare()` before consuming a stream, or use `async with engine`
to prepare on entry and close on exit. A second successful `prepare()` does nothing.
Preparation failures leave the engine unprepared and allow an explicit retry with
the same profile. Missing native components, CUDA libraries, or devices are reported
by preparation failure when the runtime provides a classifiable exception.
Preparation reads its one-second natural-speech stimulus from package data in the
installed `listenflow.faster_whisper` subtree. The stimulus is qualified under the
default `FasterWhisperSilenceMitigation` on the measured `small` and `large-v3`
CPU/`int8` and CUDA/`float16` combinations; those measurements are evidence, not a
compatibility allowlist or performance threshold. Caller-selected transcription and
silence-mitigation values pass through to warm-up unchanged, with the same upstream
outcomes and failure semantics and no retry or fallback. The engine uses no caller
path, repository fixture, model asset, or network access for the stimulus, completely
drains its segment iterator before publishing readiness, and excludes warm-up PCM and
segments from public health counters and events.

As a loading-time guide, offline `large-v3` preparation, including the one-second
speech warm-up and runtime readback, took **9.9–10.3 seconds** in three fresh Python
processes with already cached model files. The measurement used an Intel Core Ultra
5 235 CPU, four CPU threads, `device="cpu"`, requested `compute_type="int8"`
(effective `int8_float32`), beam size 1, default silence mitigation, Linux x86_64,
Python 3.13.15, faster-whisper 1.2.1, and CTranslate2 4.8.2. Model loading took
4.3–4.6 seconds and warm-up took 5.4–5.9 seconds of that total. These figures are a
guide for that configuration; storage caches, host load, model weights, decoding
settings, and runtime profile affect preparation time. The engine imposes no
deadline; callers can apply `asyncio.timeout()` and allow for native cleanup.

`FasterWhisperEngine.runtime_status()` raises `EngineStateError` until preparation
succeeds, including after a failed attempt. Once model load, packaged-speech warm-up
drain, and runtime readback all succeed, it returns an immutable
`FasterWhisperRuntimeStatus` snapshot. Requested profile and configured model-identity
fields come from the values passed to the successful load, effective device, device
indices, and compute type come from the loaded CTranslate2 model, and runtime versions
come from faster-whisper and CTranslate2; CUDA distribution metadata is included only
for an effectively CUDA-loaded model. The accessor is a fixed post-load readback, not
a compatibility check or live health probe, and its values are not embedded in
`HealthSnapshot`.

`await engine.aclose()` cancels active recognition or preparation, waits for native
work to finish, and releases the model. Repeated closure does nothing. A closed
instance cannot be prepared or used again; create a new engine. Use one engine
instance on one event loop. `health()` remains readable after closure, as defined
by the core. After closure, `capabilities` and `input_format` keep returning their
values; `runtime_status()` raises `EngineClosedError` once closure starts, including
while the engine waits for native work to finish.

### Errors

The error classes are exported from `listenflow`: `ConfigurationError` for invalid
configuration or PCM, `EngineStateError` for recognition or runtime status reads
before successful preparation or a second active stream, `EngineClosedError` for
use or interruption during engine closure, `InferenceError` for runtime failures,
`EnvironmentUnavailableError` for missing native components or identifiable CUDA
library/device failures, and `ModuleNotInstalledError` for a missing optional
runtime. They inherit from
`STTError` and the corresponding Python error type. Native and upstream exceptions
are chained as `__cause__`; the engine does not retry or substitute a profile.

### Streams and health

An engine instance permits one active stream. PCM intake and boundary tracking
continue while a closed utterance is decoded. Closed utterances are decoded once
in close order with serial runtime access, and their event groups do not interleave.
Normal input exhaustion drains all received utterances. Upstream failures reach
the caller through the core error hierarchy without retry or fallback. Close an
event iterator when stopping early, as in the example, so that cleanup completes
before the instance is reused. Caller cancellation, iterator closure, and engine
closure wait for running faster-whisper transcription to finish; faster-whisper
cannot stop it midway. Callers may use `asyncio.timeout()` to request cancellation,
but native cleanup can extend beyond that timeout. Speech yielding no text still
emits an `UtteranceStart` / `UtteranceEnd` pair.

`health()` reports `state` (`unprepared`, `ready`, or `closed`), `busy`,
`last_progress_monotonic_s`, lifetime PCM and event counters, current decode and
pending utterances, and per-session prompt
application evidence. Prompt evidence becomes applied only after a prompt-bearing
decode drains successfully and stays applied for the remainder of that session.
The progress timestamp advances on processed PCM and completed utterance decoding;
warm-up does not advance public progress. Capabilities, health, and runtime-status
reads return held values without inference or waiting for the model lock.

## Opt-in verification

The integration tests run only when their environment variables point at verified
offline model directories. The pinned model and reference-audio identities
(repository, revision, file names, and SHA-256 digests) are held in
`tests/test_faster_whisper_engine.py`. Tests verify the assets before loading or
decoding; a selected test fails when its directory does not match.

Run the tests from a checkout with the test dependencies installed. The CPU
integration run prepares and decodes both pinned models offline:

```console
uv sync
LISTENFLOW_FASTER_WHISPER_MODEL_DIR=/absolute/path/to/verified/small/model \
LISTENFLOW_FASTER_WHISPER_LARGE_V3_MODEL_DIR=/absolute/path/to/verified/large-v3/model \
uv run pytest -m 'not gpu'
```

For CUDA verification, install the extra into the uv environment and set its
library search path before starting the test process:

```console
uv sync --extra cuda
CUDA_CUBLAS_LIBRARY_DIR="$(
  uv run --extra cuda python -c \
    'from pathlib import Path; import nvidia.cublas.lib; print(Path(next(iter(nvidia.cublas.lib.__path__))))'
)"
export LD_LIBRARY_PATH="${CUDA_CUBLAS_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LISTENFLOW_FASTER_WHISPER_MODEL_DIR=/absolute/path/to/verified/small/model \
LISTENFLOW_FASTER_WHISPER_LARGE_V3_MODEL_DIR=/absolute/path/to/verified/large-v3/model \
LISTENFLOW_FASTER_WHISPER_CUDA_TEST=1 \
uv run --extra cuda pytest -m gpu
```

CUDA tests require `LISTENFLOW_FASTER_WHISPER_CUDA_TEST=1` as well as the model
directory variable for the selected model. A test whose required variables are
absent skips. Coordinate GPU use with other users of a shared host.

The language-set checks read only model metadata and need no `model.bin`:

```console
LISTENFLOW_FASTER_WHISPER_SMALL_METADATA_DIR=/absolute/path/to/small/metadata \
LISTENFLOW_FASTER_WHISPER_LARGE_V3_METADATA_DIR=/absolute/path/to/large-v3/metadata \
LISTENFLOW_FASTER_WHISPER_SMALL_EN_METADATA_DIR=/absolute/path/to/small.en/metadata \
uv run pytest -k reference_metadata
```

The reference audio `tests/fixtures/libri1_16khz_mono_s16le.wav` is tracked, and
`tests/fixtures/generate_libri1_fixture.sh` records its source and conversion.

## Versioning

Release versions are derived from exact `vX.Y.Z` tags and published on PyPI. A
checkout without a matching tag, including one with only unrelated tags, uses the
non-release fallback version `0.0.0.dev0`.

## Documentation

The [documentation guide](docs/index.md) includes the
[public API reference](docs/api/faster_whisper.md).

## License

The code is licensed under [MIT](LICENSE). The packaged warm-up recording is
provided under CC BY 4.0; its sources, attribution, and modifications are recorded
in [the speech-data notice](src/listenflow/faster_whisper/warmup_speech_attribution.txt).
