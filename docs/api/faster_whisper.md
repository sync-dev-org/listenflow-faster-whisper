# Faster-whisper API

All four classes below are exported from `listenflow.faster_whisper`. The module
also exports `__version__`, the distribution version. It exports no standalone
public functions.

## Engine

Supply an absolute `OfflineModelDirectorySource` and a positive `cpu_threads`
value. The defaults request model label `small`, language `en`, device `cpu`,
compute type `int8`, and beam size `1`. The label does not locate or verify
weights. Configuration types and ranges and metadata language support are checked
at construction; upstream decides runtime compatibility.

The constructor also accepts `max_new_tokens`, `repetition_penalty`,
`no_repeat_ngram_size`, `initial_prompt`, `word_timestamps`, `silence_mitigation`,
`boundary_settings`, and `input_format`. Decoding limits with upstream defaults
are omitted from the faster-whisper call unless changed. Audio must be 16 kHz mono
PCM, with whole samples in the configured signed 16-bit little-endian or
little-endian float32 format.

`prepare()` loads the offline model and drains the full packaged synthetic utterance
(1.36 seconds) before publishing readiness. A successful preparation is reused; an unsuccessful attempt
leaves no published partial model or status. `stream()` requires preparation and
returns an asynchronous transcript iterator. Consume only one stream per instance
and close the iterator explicitly when stopping early. Cancellation and closure
wait for uninterruptible native transcription to finish before returning.

`aclose()` cancels active operations, waits for native work, and releases the model.
Closure is idempotent and permanent. `async with engine` prepares on entry and
closes on exit. Use an instance on one event loop.

`capabilities.languages` is the accepted metadata language-code set, and
`auto_language` indicates detection for multilingual vocabularies when
`language=None`. `input_format` is the configured PCM format. `health()` supplies
`state`, `busy`, `last_progress_monotonic_s`, lifetime PCM/event counters, current
and pending decoding, and session prompt evidence; preparation warm-up does not
advance public progress. Health is readable after closure. `runtime_status()`
raises `EngineStateError` until preparation succeeds, including after a failed
attempt, then returns an immutable readback of that load.
After closure, `capabilities` and `input_format` keep returning their values;
`runtime_status()` raises `EngineClosedError` once closure starts, including while
native work is still finishing.

Configuration and PCM failures use `listenflow.ConfigurationError`; state and
closure failures use `EngineStateError` and `EngineClosedError`. Upstream failures
use `InferenceError`, identifiable missing native/CUDA components use
`EnvironmentUnavailableError`, and a missing runtime uses `ModuleNotInstalledError`.
These core errors retain native and upstream exceptions as their cause.

::: listenflow.faster_whisper.FasterWhisperEngine
    options:
      inherited_members: [prepare, stream, aclose, health, __aenter__, __aexit__]
      members: [__init__, capabilities, input_format, prepare, stream, aclose, health, runtime_status, __aenter__, __aexit__]

## Silence mitigation

This immutable configuration controls VAD and silence-related transcription
options. The same caller-selected values reach warm-up and utterance decoding.
Thresholds must be in the inclusive range `0.0` to `1.0`, silence duration must be
a nonnegative integer, and an enabled hallucination-silence threshold must be
finite and positive. Boolean fields require boolean values.

The hallucination-silence threshold is disabled by default. faster-whisper uses it
only when the engine is constructed with `word_timestamps=True`; set both to skip
long silences before likely hallucinated segments.

::: listenflow.faster_whisper.FasterWhisperSilenceMitigation
    options:
      members: [vad_filter, vad_threshold, vad_min_silence_duration_ms, no_speech_threshold, condition_on_previous_text, hallucination_silence_threshold_s, __init__]

## Runtime status

The immutable status records the configured model label and absolute directory,
requested device, compute type and CPU threads, effective device, device indices
and compute type, and faster-whisper and CTranslate2 versions. It is a fixed
post-load snapshot and does not perform live health or compatibility checks.

::: listenflow.faster_whisper.FasterWhisperRuntimeStatus

The optional CUDA status is present only when the loaded model's effective device
is CUDA. Its distribution versions are `None` when the corresponding installation
metadata is absent.

::: listenflow.faster_whisper.FasterWhisperCudaRuntimeStatus
