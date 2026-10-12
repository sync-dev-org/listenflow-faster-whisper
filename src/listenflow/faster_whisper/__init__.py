from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.resources
import math
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from listenflow.faster_whisper._version import __version__
from listenflow.faster_whisper._language import model_language_metadata, validate_language
from listenflow import (
    ConfigurationError,
    EnvironmentUnavailableError,
    InferenceError,
    ModuleNotInstalledError,
    STTError,
    UtteranceBoundarySettings,
)
from listenflow.base import STTEngineBase
from listenflow.engine import (
    ClosedUtterance,
    DiarizationMode,
    EngineCapabilities,
    EngineInputFormat,
    EngineTransport,
    FinalizationMode,
    FinalObservation,
    OfflineModelDirectorySource,
    PartialMode,
    ProvisionalObservation,
    TimestampDetail,
    UtteranceDecodePipeline,
    validate_input_format,
)
from listenflow.events import TranscriptEvent
from listenflow.errors import EngineStateError
from listenflow.health import BiasApplicationSnapshot


_REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
)
_VOCABULARY_MODEL_FILES = (
    "vocabulary.json",
    "vocabulary.txt",
)
_PREPROCESSOR_CONFIG_FILE = "preprocessor_config.json"
_BIAS_EVIDENCE_DETAIL = "faster_whisper.WhisperModel.transcribe(..., initial_prompt=...)"
_WARMUP_PCM_RESOURCE = "warmup_speech_16khz_mono_s16le.pcm"

__all__ = [
    "FasterWhisperCudaRuntimeStatus",
    "FasterWhisperEngine",
    "FasterWhisperRuntimeStatus",
    "FasterWhisperSilenceMitigation",
    "__version__",
]


def _load_faster_whisper_model() -> type[Any]:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotInstalledError(
            "FasterWhisperEngine requires the faster-whisper runtime and its dependencies. "
            "Install them with: pip install listenflow-faster-whisper."
        ) from exc
    except (ImportError, OSError) as exc:
        raise EnvironmentUnavailableError(str(exc)) from exc
    except Exception as exc:
        raise _runtime_error(exc) from exc
    return WhisperModel


def _runtime_error(error: Exception) -> STTError:
    message = str(error)
    signal = message.lower()
    missing_native_library = (
        "cannot open shared object file" in signal
        or "dll load failed" in signal
        or (".dylib" in signal and "image not found" in signal)
    )
    missing_cuda_device = any(
        detail in signal
        for detail in (
            "no cuda-capable device",
            "no cuda devices",
            "cuda driver version is insufficient",
            "invalid device ordinal",
        )
    )
    missing_cuda_library = any(
        library in signal for library in ("cublas", "cudnn", "cudart", "nvrtc", "libcuda", "nvcuda")
    ) and any(
        detail in signal
        for detail in (
            "not found",
            "cannot be loaded",
            "cannot open shared object file",
            "could not locate",
            "could not load",
            "failed to load",
        )
    )
    if isinstance(error, ModuleNotFoundError):
        return ModuleNotInstalledError(message)
    if (
        isinstance(error, ImportError)
        or missing_native_library
        or missing_cuda_device
        or missing_cuda_library
    ):
        return EnvironmentUnavailableError(message)
    if isinstance(error, (TypeError, ValueError, FileNotFoundError)):
        return ConfigurationError(message)
    return InferenceError(message)


async def _run_native(operation: Callable[..., Any], *args: object) -> Any:
    """Wait for uninterruptible native work, including on repeated cancellation."""
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - cancellation wins after native cleanup
                break
        task.exception()
        raise


def _unload_model(model: Any) -> None:
    unload = getattr(model.model, "unload_model", None)
    if unload is not None:
        unload()


def _read_warmup_pcm() -> bytes:
    return (
        importlib.resources.files("listenflow.faster_whisper")
        .joinpath(_WARMUP_PCM_RESOURCE)
        .read_bytes()
    )


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer 0 or greater")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonempty_str(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str")
    if not value:
        raise ValueError(f"{name} must be a non-empty str")


def _require_unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive")


def _require_positive_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number greater than 0")


@dataclass(frozen=True, slots=True)
class FasterWhisperSilenceMitigation:
    vad_filter: bool = True
    vad_threshold: float = 0.65
    vad_min_silence_duration_ms: int = 500
    no_speech_threshold: float = 0.75
    condition_on_previous_text: bool = False
    hallucination_silence_threshold_s: float | None = 1.0

    def __post_init__(self) -> None:
        try:
            _require_bool("vad_filter", self.vad_filter)
            _require_unit_interval("vad_threshold", self.vad_threshold)
            _require_nonnegative_int(
                "vad_min_silence_duration_ms",
                self.vad_min_silence_duration_ms,
            )
            _require_unit_interval("no_speech_threshold", self.no_speech_threshold)
            _require_bool("condition_on_previous_text", self.condition_on_previous_text)
            if self.hallucination_silence_threshold_s is not None:
                _require_positive_number(
                    "hallucination_silence_threshold_s",
                    self.hallucination_silence_threshold_s,
                )
        except (TypeError, ValueError) as error:
            raise ConfigurationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class FasterWhisperCudaRuntimeStatus:
    cublas_distribution_version: str | None
    cuda_nvrtc_distribution_version: str | None


@dataclass(frozen=True, slots=True)
class FasterWhisperRuntimeStatus:
    model_label: str
    model_directory_path: str
    requested_device: str
    effective_device: str
    device_index: tuple[int, ...]
    requested_compute_type: str
    effective_compute_type: str
    requested_cpu_threads: int
    faster_whisper_version: str
    ctranslate2_version: str
    cuda: FasterWhisperCudaRuntimeStatus | None


def _optional_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class FasterWhisperEngine(STTEngineBase):
    def __init__(
        self,
        *,
        model_source: OfflineModelDirectorySource,
        cpu_threads: int,
        model: str = "small",
        language: str | None = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        max_new_tokens: int | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        silence_mitigation: FasterWhisperSilenceMitigation = (FasterWhisperSilenceMitigation()),
        boundary_settings: UtteranceBoundarySettings = UtteranceBoundarySettings(),
        input_format: EngineInputFormat = EngineInputFormat.PCM_S16LE,
    ) -> None:
        super().__init__()
        try:
            if not isinstance(model_source, OfflineModelDirectorySource):
                raise TypeError("model_source must be OfflineModelDirectorySource")
            _require_positive_int("cpu_threads", cpu_threads)
            _require_nonempty_str("model", model)
            _require_nonempty_str("device", device)
            _require_nonempty_str("compute_type", compute_type)
            if isinstance(beam_size, bool) or not isinstance(beam_size, int):
                raise TypeError("beam_size must be a positive integer")
            if beam_size <= 0:
                raise ValueError("beam_size must be a positive integer")
            if max_new_tokens is not None:
                if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
                    raise TypeError("max_new_tokens must be None or a positive integer")
                if max_new_tokens <= 0:
                    raise ValueError("max_new_tokens must be None or a positive integer")
            if isinstance(repetition_penalty, bool) or not isinstance(
                repetition_penalty, (int, float)
            ):
                raise TypeError("repetition_penalty must be a finite number greater than 0")
            if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
                raise ValueError("repetition_penalty must be a finite number greater than 0")
            if isinstance(no_repeat_ngram_size, bool) or not isinstance(no_repeat_ngram_size, int):
                raise TypeError("no_repeat_ngram_size must be an integer 0 or greater")
            if no_repeat_ngram_size < 0:
                raise ValueError("no_repeat_ngram_size must be an integer 0 or greater")
            if initial_prompt is not None and (
                not isinstance(initial_prompt, str) or not initial_prompt
            ):
                raise ValueError("initial_prompt must be None or a non-empty string")
            _require_bool("word_timestamps", word_timestamps)
            if not isinstance(silence_mitigation, FasterWhisperSilenceMitigation):
                raise TypeError("silence_mitigation must be FasterWhisperSilenceMitigation")
            if not isinstance(boundary_settings, UtteranceBoundarySettings):
                raise TypeError("boundary_settings must be UtteranceBoundarySettings")
            input_format = validate_input_format(
                input_format,
                target="FasterWhisperEngine",
                accepted=(EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32),
            )
            accepted_languages, auto_language = model_language_metadata(model_source.directory_path)
            validate_language(language, accepted_languages)
        except (TypeError, ValueError, OSError) as error:
            raise ConfigurationError(str(error)) from error

        self.model_source = model_source
        self.cpu_threads = cpu_threads
        self.model = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        self.silence_mitigation = silence_mitigation
        self.boundary_settings = boundary_settings
        self._input_format = input_format
        self._model_cls = _load_faster_whisper_model()
        self._model: Any | None = None
        self._runtime_status: FasterWhisperRuntimeStatus | None = None
        self._session_counter = 0
        self._active_session_id: int | None = None
        self._bias_application: BiasApplicationSnapshot | None = None
        self._pipeline = UtteranceDecodePipeline(
            boundary_settings=self.boundary_settings,
            decode_utterance=self._decode_closed_utterance,
            input_format=self.input_format,
            on_progress=self._mark_progress,
        )
        self._capabilities = EngineCapabilities(
            transport=EngineTransport.IN_PROCESS,
            input_format=input_format,
            partial_mode=PartialMode.NONE,
            finalization=frozenset(
                {
                    FinalizationMode.ENGINE_SEGMENT,
                    FinalizationMode.LOCAL_VAD,
                    FinalizationMode.CLIENT_FLUSH,
                }
            ),
            timestamp_detail=TimestampDetail.SEGMENT,
            diarization=DiarizationMode.NONE,
            languages=accepted_languages,
            auto_language=auto_language,
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    @property
    def input_format(self) -> EngineInputFormat:
        return self._input_format

    async def _load(self) -> None:
        try:
            await _run_native(self._prepare_model)
        except asyncio.CancelledError:
            # Native loading cannot be stopped; _run_native waits for it first.
            await self._release()
            raise
        except STTError:
            raise
        except Exception as error:
            raise _runtime_error(error) from error

    async def _release(self) -> None:
        model = self._model
        self._model = None
        self._runtime_status = None
        if model is not None:
            await _run_native(_unload_model, model)

    def _health_details(self) -> Mapping[str, object]:
        return {
            "processed_pcm_chunks": self._pipeline.processed_pcm_chunks,
            "emitted_events": self._pipeline.emitted_events,
            "bias_application": self._bias_application,
            "decoding_utterance_id": self._pipeline.decoding_utterance_id,
            "pending_decode_utterances": self._pipeline.pending_decode_utterances,
        }

    def runtime_status(self) -> FasterWhisperRuntimeStatus:
        """Return the prepared runtime snapshot, rejecting unprepared or closed engines."""
        self._check_open()
        if not self._loaded or self._runtime_status is None:
            raise EngineStateError("prepare the engine before reading runtime status")
        return self._runtime_status

    def _prepare_model(self) -> None:
        self._validate_model_files()
        model_label = self.model
        model_directory_path = self.model_source.directory_path
        requested_device = self.device
        requested_compute_type = self.compute_type
        requested_cpu_threads = self.cpu_threads
        model = self._model_cls(
            model_directory_path,
            device=requested_device,
            device_index=0,
            compute_type=requested_compute_type,
            cpu_threads=requested_cpu_threads,
            num_workers=1,
            local_files_only=True,
        )
        try:
            self._verify_mel_bins(model)
            segments, _ = model.transcribe(
                self._pcm_to_float32(
                    _read_warmup_pcm(),
                    input_format=EngineInputFormat.PCM_S16LE,
                ),
                **self._transcribe_kwargs(),
            )
            tuple(segments)
            loaded_model = model.model
            effective_device = loaded_model.device
            device_index = tuple(loaded_model.device_index)
            effective_compute_type = loaded_model.compute_type
            faster_whisper_version = importlib.import_module("faster_whisper").__version__
            ctranslate2_version = importlib.import_module("ctranslate2").__version__
            cuda = None
            if effective_device == "cuda":
                cuda = FasterWhisperCudaRuntimeStatus(
                    cublas_distribution_version=_optional_distribution_version(
                        "nvidia-cublas-cu12"
                    ),
                    cuda_nvrtc_distribution_version=_optional_distribution_version(
                        "nvidia-cuda-nvrtc-cu12"
                    ),
                )
            runtime_status = FasterWhisperRuntimeStatus(
                model_label=model_label,
                model_directory_path=model_directory_path,
                requested_device=requested_device,
                effective_device=effective_device,
                device_index=device_index,
                requested_compute_type=requested_compute_type,
                effective_compute_type=effective_compute_type,
                requested_cpu_threads=requested_cpu_threads,
                faster_whisper_version=faster_whisper_version,
                ctranslate2_version=ctranslate2_version,
                cuda=cuda,
            )
            self._model = model
            self._runtime_status = runtime_status
        except BaseException:
            _unload_model(model)
            raise

    @staticmethod
    def _verify_mel_bins(model: Any) -> None:
        loaded_mel_bins = int(model.model.n_mels)
        configured_mel_bins = len(model.feature_extractor.mel_filters)
        if loaded_mel_bins != configured_mel_bins:
            raise ValueError(
                "model_source.directory_path feature extractor is configured for "
                f"{configured_mel_bins} mel bins but the loaded model expects "
                f"{loaded_mel_bins}: {_PREPROCESSOR_CONFIG_FILE} is missing from the "
                "directory or does not match the model"
            )

    def _validate_model_files(self) -> None:
        directory = Path(self.model_source.directory_path)
        for relative_path in _REQUIRED_MODEL_FILES:
            if not (directory / relative_path).is_file():
                raise FileNotFoundError(
                    f"model_source.directory_path is missing required file: {relative_path}"
                )
        if not any((directory / name).is_file() for name in _VOCABULARY_MODEL_FILES):
            raise FileNotFoundError(
                "model_source.directory_path is missing required file: "
                + " or ".join(_VOCABULARY_MODEL_FILES)
            )

    def _transcribe_kwargs(self) -> dict[str, object]:
        mitigation = self.silence_mitigation
        kwargs: dict[str, object] = {
            "language": "jw" if self.language == "jv" else self.language,
            "beam_size": self.beam_size,
            "initial_prompt": self.initial_prompt,
            "word_timestamps": self.word_timestamps,
            "vad_filter": mitigation.vad_filter,
            "vad_parameters": {
                "threshold": mitigation.vad_threshold,
                "min_silence_duration_ms": mitigation.vad_min_silence_duration_ms,
            },
            "no_speech_threshold": mitigation.no_speech_threshold,
            "condition_on_previous_text": mitigation.condition_on_previous_text,
            "hallucination_silence_threshold": (mitigation.hallucination_silence_threshold_s),
        }
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        if self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size != 0:
            kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        return kwargs

    async def _recognize(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        self._session_counter += 1
        session_id = self._session_counter
        self._active_session_id = session_id
        self._bias_application = BiasApplicationSnapshot(
            session_id=session_id,
            kind="prompt" if self.initial_prompt is not None else "none",
            prompt=self.initial_prompt,
            applied_value=self.initial_prompt,
            applied=False,
            evidence_method="verified_apply_path",
            evidence_detail=_BIAS_EVIDENCE_DETAIL,
        )
        pipeline_stream = self._pipeline.stream(pcm_chunks)
        try:
            async for event in pipeline_stream:
                yield event
        finally:
            try:
                await pipeline_stream.aclose()
            finally:
                self._active_session_id = None

    async def _decode_closed_utterance(
        self,
        utterance: ClosedUtterance,
    ) -> list[ProvisionalObservation | FinalObservation]:
        session_id = self._active_session_id
        bias_application = self._bias_application
        if (
            session_id is None
            or bias_application is None
            or bias_application.session_id != session_id
        ):
            raise EngineStateError("closed utterance decode has no active session")
        session_prompt = bias_application.applied_value
        try:
            segments = await _run_native(
                self._decode_utterance,
                utterance.pcm,
                session_id,
                session_prompt,
            )
        except STTError:
            raise
        except Exception as error:
            raise _runtime_error(error) from error
        observations: list[ProvisionalObservation | FinalObservation] = []
        cumulative_text: list[str] = []
        for text, segment_start_s in segments:
            cumulative_text.append(text)
            source_offset_s = utterance.start_offset_s + segment_start_s
            observations.append(
                ProvisionalObservation(
                    text=" ".join(cumulative_text),
                    source_offset_s=source_offset_s,
                )
            )
            observations.append(FinalObservation(text=text, source_offset_s=source_offset_s))
        return observations

    def _decode_utterance(
        self,
        pcm: bytes,
        session_id: int,
        session_prompt: str | None,
    ) -> list[tuple[str, float]]:
        if self._model is None:
            raise EngineStateError("FasterWhisperEngine is not prepared")
        transcribe_kwargs = self._transcribe_kwargs()
        transcribe_kwargs["initial_prompt"] = session_prompt
        runtime_segments, _ = self._model.transcribe(
            self._pcm_to_float32(pcm, input_format=self._input_format),
            **transcribe_kwargs,
        )
        decoded_segments: list[tuple[str, float]] = []
        for segment in runtime_segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            start = getattr(segment, "start", 0.0)
            decoded_segments.append((text, float(start if start is not None else 0.0)))
        if (
            self._bias_application is not None
            and self._bias_application.session_id == session_id
            and self._bias_application.applied_value == session_prompt
        ):
            self._bias_application = replace(self._bias_application, applied=True)
        return decoded_segments

    @staticmethod
    def _pcm_to_float32(
        pcm: bytes,
        *,
        input_format: EngineInputFormat,
    ) -> object:
        import numpy

        if input_format is EngineInputFormat.PCM_S16LE:
            return numpy.frombuffer(pcm, dtype="<i2").astype(numpy.float32) / 32768.0
        return numpy.frombuffer(pcm, dtype="<f4")
