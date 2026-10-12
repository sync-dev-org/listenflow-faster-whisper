from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import struct
import subprocess
import sys
import threading
import wave
from collections.abc import AsyncIterator, Iterator
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import listenflow
from listenflow import (
    BiasApplicationSnapshot,
    DiarizationMode,
    EngineInputFormat,
    EngineTransport,
    Final,
    FinalizationMode,
    ConfigurationError,
    EngineClosedError,
    EngineState,
    EngineStateError,
    EnvironmentUnavailableError,
    InferenceError,
    ModuleNotInstalledError,
    OfflineModelDirectorySource,
    Partial,
    PartialMode,
    STTEngineBase,
    TimestampDetail,
    UtteranceBoundarySettings,
    UtteranceEnd,
    UtteranceStart,
)
from listenflow.faster_whisper import FasterWhisperEngine, FasterWhisperSilenceMitigation


_PINNED_MODEL_ENV = "LISTENFLOW_FASTER_WHISPER_MODEL_DIR"
_LARGE_V3_MODEL_ENV = "LISTENFLOW_FASTER_WHISPER_LARGE_V3_MODEL_DIR"
_SMALL_METADATA_ENV = "LISTENFLOW_FASTER_WHISPER_SMALL_METADATA_DIR"
_LARGE_V3_METADATA_ENV = "LISTENFLOW_FASTER_WHISPER_LARGE_V3_METADATA_DIR"
_SMALL_EN_METADATA_ENV = "LISTENFLOW_FASTER_WHISPER_SMALL_EN_METADATA_DIR"
_CUDA_TEST_ENV = "LISTENFLOW_FASTER_WHISPER_CUDA_TEST"
_PINNED_MODEL_REPOSITORY = "Systran/faster-whisper-small"
_PINNED_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
_PINNED_DIGESTS = {
    "config.json": "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
    "model.bin": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
}
_LARGE_V3_MODEL_REPOSITORY = "Systran/faster-whisper-large-v3"
_LARGE_V3_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
_LARGE_V3_DIGESTS = {
    "config.json": "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9",
    "model.bin": "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
    "tokenizer.json": "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
    "vocabulary.json": "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
    "preprocessor_config.json": (
        "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
    ),
}
_REFERENCE_AUDIO_SIZE = 474_958
_REFERENCE_AUDIO_DIGEST = "86af93041801af297754504d195ad9b05170b667c712983972342897b93d064c"
_SMALL_EN_METADATA_DIGESTS = {
    "config.json": "666a9605530ac1f61fa8177f3702b4dacec9966749e42610839fcc32661d5fae",
    "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
    "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
}
_REFERENCE_LANGUAGE_CODES = frozenset(
    [
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "yue",
        "zh",
    ]
)


def _pcm_chunk(sample: int, *, duration_s: float) -> bytes:
    sample_count = round(16_000 * duration_s)
    return sample.to_bytes(2, byteorder="little", signed=True) * sample_count


def _format_chunk(
    input_format: EngineInputFormat,
    sample: int,
    *,
    duration_s: float,
) -> bytes:
    sample_count = round(16_000 * duration_s)
    if input_format is EngineInputFormat.PCM_S16LE:
        return struct.pack("<h", sample) * sample_count
    return struct.pack("<f", sample / 32768.0) * sample_count


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _write_model_files(directory: Path) -> OfflineModelDirectorySource:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"lang_ids": [100, 101]}))
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 100, "content": "<|en|>"},
                    {"id": 101, "content": "<|ja|>"},
                ]
            }
        )
    )
    (directory / "vocabulary.txt").write_text("token\n\n")
    (directory / "model.bin").write_bytes(b"model")
    return OfflineModelDirectorySource(directory_path=str(directory))


def _write_language_model(
    directory: Path,
    *,
    tokens: tuple[str, ...] = ("en", "ja", "jw", "iw"),
    multilingual: bool = True,
) -> OfflineModelDirectorySource:
    source = _write_model_files(directory)
    (directory / "config.json").write_text(
        json.dumps({"lang_ids": list(range(100, 100 + len(tokens)))})
    )
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": token_id, "content": f"<|{code}|>"}
                    for token_id, code in enumerate(tokens, 100)
                ]
            }
        )
    )
    (directory / "vocabulary.txt").write_text("token\n\n" if multilingual else "token\n")
    return source


def _absolute_model_path(name: str) -> str:
    return str(Path("models", name).resolve())


def _verify_pinned_model_directory(
    model_directory: Path,
    *,
    revision: str = _PINNED_REVISION,
    digests: dict[str, str] = _PINNED_DIGESTS,
) -> None:
    assert _PINNED_MODEL_REPOSITORY == "Systran/faster-whisper-small"
    assert _LARGE_V3_MODEL_REPOSITORY == "Systran/faster-whisper-large-v3"
    assert model_directory.name == revision
    assert set(digests) == {"config.json", "model.bin", "tokenizer.json"} | (
        set(digests) & {"vocabulary.json", "vocabulary.txt", "preprocessor_config.json"}
    )
    for name, expected_digest in digests.items():
        digest = hashlib.sha256((model_directory / name).read_bytes()).hexdigest()
        assert digest == expected_digest


def _verify_language_metadata(directory: Path, digests: dict[str, str]) -> None:
    assert directory.is_dir()
    for name, expected_digest in digests.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected_digest


def _assert_reference_language_set(
    directory: Path,
    expected: set[str],
    fake_runtime: type[FakeWhisperModel],
) -> None:
    for code in _REFERENCE_LANGUAGE_CODES | {"zz"}:
        source = OfflineModelDirectorySource(directory_path=str(directory))
        if code in expected:
            assert (
                FasterWhisperEngine(model_source=source, cpu_threads=1, language=code).language
                == code
            )
        else:
            with pytest.raises(ValueError, match="accepted language codes"):
                FasterWhisperEngine(model_source=source, cpu_threads=1, language=code)
    assert fake_runtime.instances == []


def _read_verified_reference_audio() -> bytes:
    fixture_path = Path(__file__).parent / "fixtures" / "libri1_16khz_mono_s16le.wav"
    assert fixture_path.stat().st_size == _REFERENCE_AUDIO_SIZE
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == _REFERENCE_AUDIO_DIGEST
    with wave.open(str(fixture_path), "rb") as fixture:
        assert fixture.getframerate() == 16_000
        assert fixture.getnchannels() == 1
        assert fixture.getsampwidth() == 2
        return fixture.readframes(fixture.getnframes())


class FakeWhisperModel:
    instances: list[FakeWhisperModel] = []
    runtime_profiles: list[object] = []

    def __init__(self, model_path: str, **kwargs: object) -> None:
        self.model_path = model_path
        self.load_kwargs = kwargs
        self.model = (
            type(self).runtime_profiles.pop(0)
            if type(self).runtime_profiles
            else SimpleNamespace(
                device="cpu",
                device_index=[0],
                compute_type="int8_float32",
                n_mels=80,
            )
        )
        self.feature_extractor = SimpleNamespace(mel_filters=[0.0] * 80)
        self.transcribe_calls: list[tuple[object, dict[str, object]]] = []
        self.responses: list[object] = []
        type(self).instances.append(self)

    def transcribe(self, audio: object, **kwargs: object) -> tuple[Iterator[object], object]:
        self.transcribe_calls.append((audio, dict(kwargs)))
        response = self.responses.pop(0) if self.responses else ()
        if isinstance(response, BaseException):

            def raise_during_drain() -> Iterator[object]:
                raise response
                yield  # pragma: no cover

            return raise_during_drain(), object()
        return iter(response), object()


class FaultInjectingWhisperModel:
    failure_site = ""
    failure: BaseException = RuntimeError("fault not configured")
    construction_calls: list[tuple[str, dict[str, object]]] = []
    transcribe_calls: list[tuple[object, dict[str, object]]] = []

    @classmethod
    def configure(cls, *, failure_site: str, failure: BaseException) -> None:
        cls.failure_site = failure_site
        cls.failure = failure
        cls.construction_calls = []
        cls.transcribe_calls = []

    def __init__(self, model_path: str, **kwargs: object) -> None:
        type(self).construction_calls.append((model_path, dict(kwargs)))
        if self.failure_site == "model construction":
            raise self.failure
        self.model = SimpleNamespace(
            device="cpu",
            device_index=[0],
            compute_type="int8_float32",
            n_mels=80,
        )
        self.feature_extractor = SimpleNamespace(mel_filters=[0.0] * 80)

    def transcribe(self, audio: object, **kwargs: object) -> tuple[Iterator[object], object]:
        type(self).transcribe_calls.append((audio, dict(kwargs)))
        call_number = len(type(self).transcribe_calls)
        call_site = "warm-up" if call_number == 1 else "utterance"
        if self.failure_site == f"{call_site} call":
            raise self.failure

        if self.failure_site == f"{call_site} drain":

            def raise_during_drain() -> Iterator[object]:
                raise self.failure
                yield  # pragma: no cover

            return raise_during_drain(), object()
        return iter(()), object()


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    FakeWhisperModel.instances = []
    FakeWhisperModel.runtime_profiles = []
    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: FakeWhisperModel,
    )
    return FakeWhisperModel


@pytest.mark.req("SP-LFFASTERWHISPER-015")
@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.parametrize(
    ("tokens", "multilingual", "accepted", "auto_language"),
    [
        (("ja", "jw"), False, frozenset({"en"}), False),
        (("en", "ja", "jw", "iw"), True, frozenset({"en", "ja", "jv"}), True),
        (("en", "haw", "yue"), True, frozenset({"en", "haw", "yue"}), True),
        (("en",), True, frozenset({"en"}), True),
    ],
)
def test_declared_languages_equal_the_values_accepted_before_recognition(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    tokens: tuple[str, ...],
    multilingual: bool,
    accepted: frozenset[str],
    auto_language: bool,
) -> None:
    """Capabilities declare exactly the model's accepted codes and its language-detection support."""

    source = _write_language_model(tmp_path / "model", tokens=tokens, multilingual=multilingual)
    for value in (*sorted(accepted), None):
        engine = FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
        assert engine.language == value
        assert engine.capabilities.languages == accepted
        assert engine.capabilities.auto_language is auto_language
    for value in ("EN", "English", "en-US", "en_us", "eng", "iw", "xx", 7):
        with pytest.raises(ConfigurationError, match="language") as captured:
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
        assert isinstance(captured.value.__cause__, (TypeError, ValueError))
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_english_only_vocabulary_accepts_only_english(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """An English-only vocabulary accepts en even when its language IDs list other codes."""

    source = _write_language_model(
        tmp_path / "model", tokens=("ja", "jw", "iw"), multilingual=False
    )
    assert FasterWhisperEngine(model_source=source, cpu_threads=1, language="en").language == "en"
    for value in ("ja", "Japanese"):
        with pytest.raises(ConfigurationError, match="accepted language codes: en") as error:
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
        assert "use 'ja'" not in str(error.value)
    for value in ("English", "EN"):
        with pytest.raises(ConfigurationError, match="use 'en'"):
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_multilingual_vocabulary_accepts_only_usable_listed_language_tokens(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """A multilingual model accepts its usable listed codes, excluding absent and unsupported tokens."""

    source = _write_language_model(tmp_path / "model")
    for code in ("en", "ja", "jv"):
        assert (
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=code).language == code
        )
    for code in ("yue", "he", "iw"):
        with pytest.raises(
            ConfigurationError, match="accepted language codes: en, ja, jv"
        ) as error:
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=code)
        assert f"use '{code}'" not in str(error.value)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_vocabulary_json_takes_precedence_over_vocabulary_txt(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The vocabulary JSON file decides English-only status when both vocabulary files exist."""

    source = _write_language_model(tmp_path / "model")
    directory = Path(source.directory_path)
    (directory / "vocabulary.json").write_text(json.dumps(["token"]))
    with pytest.raises(ConfigurationError, match="accepted language codes: en"):
        FasterWhisperEngine(model_source=source, cpu_threads=1, language="ja")
    (directory / "vocabulary.txt").write_text("token\n")
    (directory / "vocabulary.json").write_text(json.dumps(["token", ""]))
    assert FasterWhisperEngine(model_source=source, cpu_threads=1, language="ja").language == "ja"
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_unusable_language_metadata_is_rejected_at_construction(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The engine rejects metadata that cannot establish the languages a multilingual model handles."""

    source = _write_language_model(tmp_path / "model")
    directory = Path(source.directory_path)
    (directory / "vocabulary.txt").unlink()
    with pytest.raises(ConfigurationError, match=r"vocabulary\.json or vocabulary\.txt"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    (directory / "vocabulary.txt").write_text("token\n\n")
    for config in ({}, {"lang_ids": []}, {"lang_ids": [999]}):
        (directory / "config.json").write_text(json.dumps(config))
        with pytest.raises(ConfigurationError, match="lang_ids"):
            FasterWhisperEngine(model_source=source, cpu_threads=1)
    (directory / "config.json").write_text(json.dumps({"lang_ids": [100]}))
    (directory / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 100, "content": "<|transcribe|>"}]})
    )
    with pytest.raises(ConfigurationError, match="lang_ids"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_broken_language_metadata_is_rejected_at_construction(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Unreadable language metadata fails at construction before any runtime model load."""

    source = _write_language_model(tmp_path / "model")
    directory = Path(source.directory_path)
    for name in ("config.json", "tokenizer.json"):
        path = directory / name
        original = path.read_bytes()
        path.write_text("{")
        with pytest.raises(ConfigurationError, match=name):
            FasterWhisperEngine(model_source=source, cpu_threads=1)
        path.write_bytes(original)
    (directory / "vocabulary.json").write_text("{")
    with pytest.raises(ConfigurationError, match="vocabulary.json"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_text_vocabulary_uses_ctranslate2_line_endings(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The model's English-only status follows CTranslate2's text vocabulary line handling."""

    source = _write_language_model(tmp_path / "model")
    vocabulary = Path(source.directory_path) / "vocabulary.txt"
    cases = (
        (b"token\n\r\nother\n", False),
        (b"token\r\n\r\n", True),
        (b"token\n\n", True),
        (b"token\x0b\x0bother\n", False),
    )
    for content, multilingual in cases:
        vocabulary.write_bytes(content)
        if multilingual:
            assert (
                FasterWhisperEngine(model_source=source, cpu_threads=1, language="ja").language
                == "ja"
            )
        else:
            with pytest.raises(ConfigurationError, match="accepted language codes: en"):
                FasterWhisperEngine(model_source=source, cpu_threads=1, language="ja")
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-004")
def test_non_file_metadata_entries_fail_at_construction(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Metadata paths that are not regular files fail at construction with their filenames."""

    source = _write_language_model(tmp_path / "model")
    directory = Path(source.directory_path)
    text_vocabulary = directory / "vocabulary.txt"
    text_vocabulary.unlink()
    json_vocabulary = directory / "vocabulary.json"
    json_vocabulary.mkdir()
    with pytest.raises(ConfigurationError, match="vocabulary.json"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    text_vocabulary.write_text("token\n\n")
    with pytest.raises(ConfigurationError, match="vocabulary.json"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    json_vocabulary.rmdir()
    for name in ("config.json", "tokenizer.json"):
        path = directory / name
        original = path.read_bytes()
        path.unlink()
        path.mkdir()
        with pytest.raises(ConfigurationError, match=name):
            FasterWhisperEngine(model_source=source, cpu_threads=1)
        path.rmdir()
        path.write_bytes(original)
    (directory / "vocabulary-target.json").write_text(json.dumps(["token", ""]))
    json_vocabulary.symlink_to("vocabulary-target.json")
    assert FasterWhisperEngine(model_source=source, cpu_threads=1, language="ja").language == "ja"
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_added_tokens_with_a_bad_shape_is_a_metadata_value_error(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Malformed tokenizer added tokens are rejected as configuration values."""

    source = _write_language_model(tmp_path / "model")
    tokenizer = Path(source.directory_path) / "tokenizer.json"
    for metadata in ({}, {"added_tokens": None}, {"added_tokens": {}}):
        tokenizer.write_text(json.dumps(metadata))
        with pytest.raises(ConfigurationError, match="tokenizer.json added_tokens"):
            FasterWhisperEngine(model_source=source, cpu_threads=1)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
def test_language_error_hints_cover_only_languages_the_model_handles(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Language errors suggest a contract code only when the model handles that language."""

    source = _write_language_model(tmp_path / "model")
    for value in ("English", "EN", "en-US", "en_us", "eng"):
        with pytest.raises(ConfigurationError, match="use 'en'"):
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
    for value in ("Japanese", "JA", "ja-JP", "jpn"):
        with pytest.raises(ConfigurationError, match="use 'ja'"):
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
    for value in ("Hawaiian", "haw", "Cantonese", "yue", "Klingon"):
        with pytest.raises(
            ConfigurationError, match="accepted language codes: en, ja, jv"
        ) as error:
            FasterWhisperEngine(model_source=source, cpu_threads=1, language=value)
        assert "use '" not in str(error.value)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
@pytest.mark.asyncio
async def test_javanese_contract_code_is_passed_to_upstream_as_jw(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The engine maps accepted Javanese jv to upstream jw for transcription."""

    source = _write_language_model(tmp_path / "model")
    engine = FasterWhisperEngine(model_source=source, cpu_threads=1, language="jv")
    await engine.prepare()
    assert engine.language == "jv"
    assert fake_runtime.instances[0].transcribe_calls[0][1]["language"] == "jw"


@pytest.mark.req("SP-LFFASTERWHISPER-002")
@pytest.mark.req("SP-LFFASTERWHISPER-015")
@pytest.mark.asyncio
async def test_none_language_is_forwarded_for_detection_or_english(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """None leaves the language unspecified for multilingual detection and English-only transcription."""

    assert get_type_hints(FasterWhisperEngine.__init__)["language"] == str | None
    for name, multilingual in (("multi", True), ("english", False)):
        source = _write_language_model(tmp_path / name, multilingual=multilingual)
        engine = FasterWhisperEngine(model_source=source, cpu_threads=1, language=None)
        await engine.prepare()
        assert engine.language is None
        assert fake_runtime.instances[-1].transcribe_calls[0][1]["language"] is None


@pytest.mark.req("SP-LFFASTERWHISPER-004")
def test_missing_model_directory_is_reported_at_construction(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """A missing offline model directory fails at construction before model loading."""

    source = OfflineModelDirectorySource(directory_path=str(tmp_path / "missing"))
    with pytest.raises(ConfigurationError, match="directory_path"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-015")
@pytest.mark.skipif(
    not os.environ.get(_SMALL_METADATA_ENV),
    reason=f"set {_SMALL_METADATA_ENV} to check the pinned small language metadata",
)
def test_small_reference_metadata_accepts_its_99_languages(
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The verified small conversion accepts its 99 language tokens and excludes yue."""

    directory = Path(os.environ[_SMALL_METADATA_ENV])
    _verify_language_metadata(
        directory, {name: digest for name, digest in _PINNED_DIGESTS.items() if name != "model.bin"}
    )
    expected = set(_REFERENCE_LANGUAGE_CODES - {"yue"})
    assert len(expected) == 99
    _assert_reference_language_set(directory, expected, fake_runtime)


@pytest.mark.req("SP-LFFASTERWHISPER-015")
@pytest.mark.skipif(
    not os.environ.get(_LARGE_V3_METADATA_ENV),
    reason=f"set {_LARGE_V3_METADATA_ENV} to check the pinned large-v3 language metadata",
)
def test_large_v3_reference_metadata_accepts_its_100_languages(
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The verified large-v3 conversion accepts its 100 usable language tokens from vocabulary JSON."""

    directory = Path(os.environ[_LARGE_V3_METADATA_ENV])
    _verify_language_metadata(
        directory,
        {name: digest for name, digest in _LARGE_V3_DIGESTS.items() if name != "model.bin"},
    )
    expected = set(_REFERENCE_LANGUAGE_CODES)
    assert len(expected) == 100
    _assert_reference_language_set(directory, expected, fake_runtime)


@pytest.mark.req("SP-LFFASTERWHISPER-015")
@pytest.mark.skipif(
    not os.environ.get(_SMALL_EN_METADATA_ENV),
    reason=f"set {_SMALL_EN_METADATA_ENV} to check the pinned small.en language metadata",
)
def test_small_en_reference_metadata_accepts_only_english(
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The verified small.en conversion accepts only en despite its other language IDs."""

    directory = Path(os.environ[_SMALL_EN_METADATA_ENV])
    _verify_language_metadata(directory, _SMALL_EN_METADATA_DIGESTS)
    expected = {"en"}
    _assert_reference_language_set(directory, expected, fake_runtime)


async def _prepared_engine(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    **kwargs: object,
) -> tuple[FasterWhisperEngine, FakeWhisperModel]:
    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(model_source=source, cpu_threads=3, **kwargs)
    await engine.prepare()
    model = fake_runtime.instances[-1]
    model.transcribe_calls.clear()
    return engine, model


class _ControlledRuntime:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        responses: list[object],
        blocked_calls: set[int],
    ) -> None:
        self._loop = loop
        self._responses = responses
        self._lock = threading.Lock()
        self.started = [asyncio.Event() for _ in responses]
        self.releases = [threading.Event() for _ in responses]
        for index, release in enumerate(self.releases):
            if index not in blocked_calls:
                release.set()
        self.drained = [False for _ in responses]
        self.finished = [asyncio.Event() for _ in responses]
        self.audio: list[tuple[float, ...]] = []
        self.kwargs: list[dict[str, object]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    def transcribe(self, audio: object, **kwargs: object) -> tuple[Iterator[object], object]:
        call_index = len(self.audio)
        self.audio.append(tuple(float(sample) for sample in audio))
        self.kwargs.append(dict(kwargs))
        response = self._responses[call_index] if call_index < len(self._responses) else ()
        if call_index >= len(self.started):
            self.started.append(asyncio.Event())
            release = threading.Event()
            release.set()
            self.releases.append(release)
            self.drained.append(False)
            self.finished.append(asyncio.Event())

        def iterate() -> Iterator[object]:
            with self._lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self._loop.call_soon_threadsafe(self.started[call_index].set)
            try:
                self.releases[call_index].wait()
                if isinstance(response, BaseException):
                    raise response
                yield from response
                self.drained[call_index] = True
            finally:
                with self._lock:
                    self.active_calls -= 1
                self._loop.call_soon_threadsafe(self.finished[call_index].set)

        return iterate(), object()


class _GatedChunks:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        gate: asyncio.Event | None = None,
        gate_before_index: int | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._gate = gate
        self._gate_before_index = gate_before_index
        self._failure = failure
        self._index = 0
        self.yielded: list[bytes] = []
        self.exhaustion_requested = asyncio.Event()
        self.failure_raised = asyncio.Event()

    def __aiter__(self) -> _GatedChunks:
        return self

    async def __anext__(self) -> bytes:
        if self._index == self._gate_before_index and self._gate is not None:
            await self._gate.wait()
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            self.yielded.append(chunk)
            return chunk
        if self._failure is not None:
            self.failure_raised.set()
            raise self._failure
        self.exhaustion_requested.set()
        raise StopAsyncIteration


async def _run_ready_callbacks(*, turns: int = 20) -> None:
    loop = asyncio.get_running_loop()
    for _ in range(turns):
        checkpoint = loop.create_future()
        loop.call_soon(checkpoint.set_result, None)
        await checkpoint


async def _collect_events(events: AsyncIterator[object]) -> list[object]:
    return [event async for event in events]


async def _collect_events_into(
    events: AsyncIterator[object],
    collected: list[object],
) -> None:
    async for event in events:
        collected.append(event)


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.parametrize("termination", ["caller cancellation", "engine close"])
async def test_interrupted_preparation_finishes_native_work_and_releases_its_resources(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    termination: str,
) -> None:
    """Interrupted preparation waits for its native warm-up and returns every resource it created."""

    control = _ControlledRuntime(loop=asyncio.get_running_loop(), responses=[()], blocked_calls={0})
    released: list[bool] = []

    class Model(fake_runtime):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.model.unload_model = lambda: released.append(control.drained[0])
            self.transcribe = control.transcribe

    engine = FasterWhisperEngine(model_source=_write_model_files(tmp_path / "model"), cpu_threads=1)
    engine._model_cls = Model
    preparing = asyncio.create_task(engine.prepare())
    await control.started[0].wait()
    if termination == "caller cancellation":
        preparing.cancel()
        stopping = preparing
    else:
        stopping = asyncio.create_task(engine.aclose())
    try:
        await _run_ready_callbacks()
        assert not stopping.done()
        assert not control.finished[0].is_set()
        assert released == []
        with pytest.raises(
            EngineClosedError if termination == "engine close" else EngineStateError
        ):
            engine.runtime_status()
    finally:
        control.releases[0].set()
        if termination == "caller cancellation":
            with pytest.raises(asyncio.CancelledError):
                await stopping
        else:
            await stopping
            with pytest.raises(EngineClosedError):
                await preparing
        await control.finished[0].wait()
    assert released == [True]
    assert engine.health().busy is False
    if termination == "caller cancellation":
        assert engine.health().state is EngineState.UNPREPARED
        with pytest.raises(EngineStateError):
            engine.runtime_status()
        engine._model_cls = fake_runtime
        await engine.prepare()
        assert engine.health().state is EngineState.READY
    else:
        assert engine.health().state is EngineState.CLOSED
        with pytest.raises(EngineClosedError):
            engine.runtime_status()
    await engine.aclose()


@pytest.mark.req("SP-LFFASTERWHISPER-014")
async def test_engine_context_prepares_once_and_closes_the_instance_permanently(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Closure preserves settings and refuses runtime status, preparation, and recognition."""

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
        input_format=EngineInputFormat.PCM_F32,
    )
    capabilities = engine.capabilities
    input_format = engine.input_format
    with pytest.raises(EngineStateError) as unprepared:
        engine.runtime_status()
    assert type(unprepared.value) is EngineStateError
    with pytest.raises(EngineStateError):
        await anext(engine.stream(_chunks()))
    assert fake_runtime.instances == []
    releases = []
    async with engine as entered:
        assert entered is engine
        assert engine.health().state is EngineState.READY
        assert engine.runtime_status() is not None
        assert not hasattr(engine, "preflight")
        fake_runtime.instances[0].model.unload_model = lambda: releases.append(True)
        await engine.prepare()
        assert len(fake_runtime.instances) == 1
    assert engine.health().state is EngineState.CLOSED
    assert engine.health().busy is False
    assert releases == [True]
    assert engine.capabilities == capabilities
    assert engine.input_format is input_format is EngineInputFormat.PCM_F32
    with pytest.raises(EngineClosedError):
        engine.runtime_status()
    with pytest.raises(EngineClosedError):
        await engine.prepare()
    with pytest.raises(EngineClosedError):
        await anext(engine.stream(_chunks()))
    await engine.aclose()
    assert releases == [True]


class _RaisingRuntimeReadback:
    def __init__(self, field: str, failure: BaseException) -> None:
        self._field = field
        self._failure = failure

    def __getattribute__(self, name: str) -> object:
        if name in {"device", "device_index", "compute_type", "n_mels"}:
            if name == object.__getattribute__(self, "_field"):
                raise object.__getattribute__(self, "_failure")
            return {
                "device": "cpu",
                "device_index": [0],
                "compute_type": "int8_float32",
                "n_mels": 80,
            }[name]
        return object.__getattribute__(self, name)


class _FailOncePreparationModel:
    failure_site = ""
    failure: BaseException = RuntimeError("fault not configured")
    construction_calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def configure(cls, *, failure_site: str, failure: BaseException) -> None:
        cls.failure_site = failure_site
        cls.failure = failure
        cls.construction_calls = []

    def __init__(self, model_path: str, **kwargs: object) -> None:
        type(self).construction_calls.append((model_path, dict(kwargs)))
        self.attempt = len(type(self).construction_calls)
        if self.attempt == 1 and self.failure_site == "model construction":
            raise self.failure
        self.model = SimpleNamespace(
            device="cpu",
            device_index=[0],
            compute_type="int8_float32",
            n_mels=80,
        )
        self.feature_extractor = SimpleNamespace(mel_filters=[0.0] * 80)

    def transcribe(self, _audio: object, **_kwargs: object) -> tuple[Iterator[object], object]:
        if self.attempt == 1 and self.failure_site == "warm-up call":
            raise self.failure
        if self.attempt == 1 and self.failure_site == "warm-up drain":

            def raise_during_drain() -> Iterator[object]:
                raise self.failure
                yield  # pragma: no cover

            return raise_during_drain(), object()
        return iter(()), object()


class _FailOnceVersionModule:
    def __init__(self, wrapped: object, failure: BaseException) -> None:
        self._wrapped = wrapped
        self._failure = failure
        self._failed = False

    def __getattr__(self, name: str) -> object:
        if name == "__version__" and not self._failed:
            self._failed = True
            raise self._failure
        return getattr(self._wrapped, name)


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-009")
@pytest.mark.parametrize("preparation", ["explicit", "context"])
async def test_ac_12_1_voiced_warmup_drains_before_readiness_without_public_progress(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    preparation: str,
) -> None:
    """Preparation drains voiced packaged speech before publishing readiness without counting public activity."""

    overrides = (
        {
            "language": "ja",
            "beam_size": 5,
            "initial_prompt": "listen flow",
            "word_timestamps": True,
            "silence_mitigation": FasterWhisperSilenceMitigation(
                vad_filter=False,
                vad_threshold=0.4,
                vad_min_silence_duration_ms=250,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                hallucination_silence_threshold_s=None,
            ),
        }
        if preparation == "context"
        else {}
    )
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
        **overrides,
    )
    before = engine.health()
    phases: list[str] = []
    drained = False

    def observe_unready(phase: str) -> None:
        with pytest.raises(EngineStateError):
            engine.runtime_status()
        assert engine.health().state is EngineState.UNPREPARED
        phases.append(phase)

    class Readback:
        n_mels = 80

        def __getattr__(self, name: str) -> object:
            observe_unready("readback")
            assert drained
            return {"device": "cpu", "device_index": [0], "compute_type": "int8"}[name]

    class RecordingModel(fake_runtime):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.model = Readback()
            observe_unready("load")

        def transcribe(self, audio: object, **kwargs: object) -> tuple[Iterator[object], object]:
            if self.transcribe_calls:
                assert drained
                assert engine.runtime_status() is not None
                assert engine.health().state is EngineState.READY
                return super().transcribe(audio, **kwargs)
            self.transcribe_calls.append((audio, kwargs))
            observe_unready("warm-up call")

            def segments() -> Iterator[object]:
                nonlocal drained
                for text in ("warm", "up"):
                    observe_unready("warm-up segment")
                    yield SimpleNamespace(text=text, start=0.0)
                observe_unready("warm-up exhausted")
                drained = True

            return segments(), object()

    engine._model_cls = RecordingModel
    if preparation == "explicit":
        await engine.prepare()
        assert engine.health() == replace(before, state=EngineState.READY)
    else:
        await engine.__aenter__()
        events = [
            event
            async for event in engine.stream(
                _chunks(
                    _pcm_chunk(700, duration_s=0.01),
                    _pcm_chunk(0, duration_s=0.1),
                    _pcm_chunk(800, duration_s=0.01),
                )
            )
        ]
        assert [type(event) for event in events] == [UtteranceStart, UtteranceEnd] * 2
        assert engine.health().processed_pcm_chunks == 3
        assert engine.health().emitted_events == 4

    model = fake_runtime.instances[-1]
    assert len(model.transcribe_calls) == (1 if preparation == "explicit" else 3)
    if preparation == "context":
        for _, kwargs in model.transcribe_calls:
            assert kwargs == {
                "language": "ja",
                "beam_size": 5,
                "initial_prompt": "listen flow",
                "word_timestamps": True,
                "vad_filter": False,
                "vad_parameters": {"threshold": 0.4, "min_silence_duration_ms": 250},
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "hallucination_silence_threshold": None,
            }
    assert phases[:5] == [
        "load",
        "warm-up call",
        "warm-up segment",
        "warm-up segment",
        "warm-up exhausted",
    ]
    assert phases[5:] and set(phases[5:]) == {"readback"}
    assert drained
    assert engine.runtime_status() is not None
    assert engine.health().state is EngineState.READY
    audio, _ = model.transcribe_calls[0]
    assert any(audio), "preparation input must contain speech, not digital silence"
    assert len(audio) == 16_000, "the one-second preparation clip is interpreted at 16 kHz"


@pytest.mark.req("SP-LFFASTERWHISPER-002")
@pytest.mark.asyncio
async def test_ac_6_1_runtime_status_public_shape_and_immutability(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The public runtime status exposes the specified fields as immutable module-owned values."""

    module = listenflow.faster_whisper
    runtime_status_type = getattr(module, "FasterWhisperRuntimeStatus", None)
    cuda_status_type = getattr(module, "FasterWhisperCudaRuntimeStatus", None)

    assert runtime_status_type is not None
    assert cuda_status_type is not None
    assert is_dataclass(runtime_status_type)
    assert is_dataclass(cuda_status_type)
    assert [field.name for field in fields(runtime_status_type)] == [
        "model_label",
        "model_directory_path",
        "requested_device",
        "effective_device",
        "device_index",
        "requested_compute_type",
        "effective_compute_type",
        "requested_cpu_threads",
        "faster_whisper_version",
        "ctranslate2_version",
        "cuda",
    ]
    assert get_type_hints(runtime_status_type) == {
        "model_label": str,
        "model_directory_path": str,
        "requested_device": str,
        "effective_device": str,
        "device_index": tuple[int, ...],
        "requested_compute_type": str,
        "effective_compute_type": str,
        "requested_cpu_threads": int,
        "faster_whisper_version": str,
        "ctranslate2_version": str,
        "cuda": cuda_status_type | None,
    }
    assert [field.name for field in fields(cuda_status_type)] == [
        "cublas_distribution_version",
        "cuda_nvrtc_distribution_version",
    ]
    assert get_type_hints(cuda_status_type) == {
        "cublas_distribution_version": str | None,
        "cuda_nvrtc_distribution_version": str | None,
    }
    assert "FasterWhisperRuntimeStatus" in module.__all__
    assert "FasterWhisperCudaRuntimeStatus" in module.__all__
    assert "FasterWhisperRuntimeStatus" not in listenflow.__all__
    assert "FasterWhisperCudaRuntimeStatus" not in listenflow.__all__

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
    )
    await engine.prepare()
    status = engine.runtime_status()

    assert status is not None
    assert isinstance(status.device_index, tuple)
    with pytest.raises(FrozenInstanceError):
        status.model_label = "replacement"


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
async def test_ac_6_2_status_is_published_only_after_success_and_is_fixed(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Successful preparation publishes a fixed runtime status; repeated preparation keeps it unchanged."""

    source = _write_model_files(tmp_path / "model")
    explicit = FasterWhisperEngine(model_source=source, cpu_threads=3)

    with pytest.raises(EngineStateError):
        explicit.runtime_status()
    assert explicit.health().state is EngineState.UNPREPARED

    await explicit.prepare()
    first = explicit.runtime_status()
    assert first is not None
    assert explicit.health().state is EngineState.READY

    explicit.device = "cuda"
    explicit.compute_type = "float16"
    explicit.cpu_threads = 99
    await explicit.prepare()

    assert explicit.runtime_status() == first

    async with FasterWhisperEngine(model_source=source, cpu_threads=3) as contextual:
        events = [event async for event in contextual.stream(_chunks())]
        assert events == []
        assert contextual.runtime_status() == first
        assert contextual.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["device", "device_index", "compute_type"])
async def test_ac_6_2_loaded_runtime_readback_failure_is_atomic_and_retryable(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    field: str,
) -> None:
    """A loaded-model readback failure leaves preparation unready and permits a later retry."""

    expected = RuntimeError(f"cannot read {field}")
    fake_runtime.runtime_profiles = [
        _RaisingRuntimeReadback(field, expected),
        SimpleNamespace(device="cpu", device_index=[0], compute_type="int8_float32", n_mels=80),
    ]
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
    )

    with pytest.raises(RuntimeError) as captured:
        await engine.prepare()

    assert captured.value.__cause__ is expected
    with pytest.raises(EngineStateError):
        engine.runtime_status()
    assert engine.health().state is EngineState.UNPREPARED

    await engine.prepare()

    assert engine.runtime_status() is not None
    assert engine.health().state is EngineState.READY
    assert len(fake_runtime.instances) == 2


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
@pytest.mark.parametrize("module_name", ["faster_whisper", "ctranslate2"])
async def test_ac_6_2_runtime_version_failure_is_atomic_and_retryable(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    """A runtime version read failure leaves preparation unready and permits a later retry."""

    expected = RuntimeError(f"cannot read {module_name} version")
    wrapped = importlib.import_module(module_name)
    monkeypatch.setitem(sys.modules, module_name, _FailOnceVersionModule(wrapped, expected))
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
    )

    with pytest.raises(RuntimeError) as captured:
        await engine.prepare()

    assert captured.value.__cause__ is expected
    with pytest.raises(EngineStateError):
        engine.runtime_status()
    assert engine.health().state is EngineState.UNPREPARED

    await engine.prepare()

    assert engine.runtime_status() is not None
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_site", ["model construction", "warm-up call", "warm-up drain"])
async def test_ac_6_2_failed_load_or_warm_up_can_retry_without_partial_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    """A failed load or warm-up exposes no partial readiness and permits a later retry."""

    expected = RuntimeError(f"injected {failure_site}")
    _FailOncePreparationModel.configure(failure_site=failure_site, failure=expected)
    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: _FailOncePreparationModel,
    )
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
    )

    with pytest.raises(RuntimeError) as captured:
        await engine.prepare()

    assert captured.value.__cause__ is expected
    with pytest.raises(EngineStateError):
        engine.runtime_status()
    assert engine.health().state is EngineState.UNPREPARED

    await engine.prepare()

    assert engine.runtime_status() is not None
    assert engine.health().state is EngineState.READY
    assert len(_FailOncePreparationModel.construction_calls) == 2


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.parametrize("input_format", [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32])
async def test_unprepared_recognition_is_rejected_without_reading_pcm_or_loading(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
) -> None:
    """Recognition requires explicit preparation and leaves the caller's input untouched on rejection."""

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
        input_format=input_format,
    )
    source = _GatedChunks([_format_chunk(input_format, 700, duration_s=0.01)])
    with pytest.raises(EngineStateError, match="prepare"):
        await anext(engine.stream(source))
    assert source.yielded == []
    assert fake_runtime.instances == []
    with pytest.raises(EngineStateError):
        engine.runtime_status()
    before = engine.health()
    assert before.state is EngineState.UNPREPARED
    assert before.busy is False
    assert before.processed_pcm_chunks == 0 and before.emitted_events == 0
    assert before.last_progress_monotonic_s is None
    await engine.prepare()
    events = [event async for event in engine.stream(source)]
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.01),
    ]
    await engine.aclose()


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
async def test_ac_6_2_unexpected_cuda_metadata_failure_is_atomic_and_retryable(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected CUDA metadata failure leaves preparation unready and permits a later retry."""

    expected = RuntimeError("metadata backend failed")
    real_version = importlib.metadata.version
    fake_runtime.runtime_profiles = [
        SimpleNamespace(device="cuda", device_index=[0], compute_type="float16", n_mels=80),
        SimpleNamespace(device="cuda", device_index=[0], compute_type="float16", n_mels=80),
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(expected),
    )
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
        device="cuda",
        compute_type="float16",
    )

    with pytest.raises(RuntimeError) as captured:
        await engine.prepare()

    assert captured.value.__cause__ is expected
    with pytest.raises(EngineStateError):
        engine.runtime_status()
    assert engine.health().state is EngineState.UNPREPARED

    monkeypatch.setattr(importlib.metadata, "version", real_version)
    await engine.prepare()

    assert engine.runtime_status() is not None
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
async def test_ac_6_3_requested_and_effective_cpu_profiles_remain_distinct(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Runtime status keeps the requested CPU profile distinct from the loaded effective profile."""

    default = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "default"),
        cpu_threads=3,
    )
    await default.prepare()

    default_status = default.runtime_status()
    assert default_status is not None
    assert (
        default_status.requested_device,
        default_status.effective_device,
        default_status.device_index,
        default_status.requested_compute_type,
        default_status.effective_compute_type,
    ) == ("cpu", "cpu", (0,), "int8", "int8_float32")

    changed = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "changed"),
        cpu_threads=1,
        device="auto",
        compute_type="auto",
    )
    changed.device = "cpu"
    changed.compute_type = "default"
    changed.cpu_threads = 7
    await changed.prepare()

    changed_status = changed.runtime_status()
    assert changed_status is not None
    assert changed_status.requested_device == "cpu"
    assert changed_status.requested_compute_type == "default"
    assert changed_status.requested_cpu_threads == 7
    assert fake_runtime.instances[-1].load_kwargs["device"] == "cpu"
    assert fake_runtime.instances[-1].load_kwargs["compute_type"] == "default"
    assert fake_runtime.instances[-1].load_kwargs["cpu_threads"] == 7


@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
async def test_ac_6_4_model_identity_is_the_exact_configured_label_and_directory(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Runtime status records the exact caller-supplied model label and directory without inferred identity."""

    source = _write_model_files(tmp_path / "model")
    engines = [
        FasterWhisperEngine(model_source=source, cpu_threads=1, model=model)
        for model in ("small", "private-conversion")
    ]

    for engine in engines:
        await engine.prepare()

    statuses = [engine.runtime_status() for engine in engines]
    assert [status.model_label for status in statuses if status is not None] == [
        "small",
        "private-conversion",
    ]
    assert all(
        status is not None and status.model_directory_path == source.directory_path
        for status in statuses
    )
    assert all(
        {field.name for field in fields(status)}.isdisjoint({"model_name", "model_digest"})
        for status in statuses
        if status is not None
    )


@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
async def test_ac_6_5_cpu_status_uses_runtime_versions_without_cuda_metadata_probe(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU runtime status reports loaded runtime versions without querying CUDA distribution metadata."""

    faster_whisper_runtime = importlib.import_module("faster_whisper")
    ctranslate2_runtime = importlib.import_module("ctranslate2")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(AssertionError("CPU must not query CUDA metadata")),
    )
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
        device="cuda",
    )

    await engine.prepare()

    status = engine.runtime_status()
    assert status is not None
    assert status.effective_device == "cpu"
    assert status.faster_whisper_version == faster_whisper_runtime.__version__
    assert status.ctranslate2_version == ctranslate2_runtime.__version__
    assert status.cuda is None


@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missing", "expected_versions"),
    [
        (set(), ("12.9.2.10", "12.9.86")),
        ({"nvidia-cublas-cu12"}, (None, "12.9.86")),
        ({"nvidia-cuda-nvrtc-cu12"}, ("12.9.2.10", None)),
        ({"nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu12"}, (None, None)),
    ],
)
async def test_ac_6_8_cuda_status_preserves_available_distribution_metadata(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
    missing: set[str],
    expected_versions: tuple[str | None, str | None],
) -> None:
    """CUDA runtime status preserves available distribution versions and marks missing metadata as unavailable."""

    versions = {
        "nvidia-cublas-cu12": "12.9.2.10",
        "nvidia-cuda-nvrtc-cu12": "12.9.86",
    }

    def distribution_version(name: str) -> str:
        if name in missing:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", distribution_version)
    fake_runtime.runtime_profiles = [
        SimpleNamespace(device="cuda", device_index=[0], compute_type="float16", n_mels=80)
    ]
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
        device="cpu",
        compute_type="auto",
    )

    await engine.prepare()

    status = engine.runtime_status()
    assert status is not None
    assert status.requested_device == "cpu"
    assert status.effective_device == "cuda"
    assert status.requested_compute_type == "auto"
    assert status.effective_compute_type == "float16"
    assert isinstance(status.cuda, listenflow.faster_whisper.FasterWhisperCudaRuntimeStatus)
    assert (
        status.cuda.cublas_distribution_version,
        status.cuda.cuda_nvrtc_distribution_version,
    ) == expected_versions
    with pytest.raises(FrozenInstanceError):
        status.cuda.cublas_distribution_version = "replacement"


@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_device", "effective_device", "has_cuda"),
    [("cpu", "cuda", True), ("cuda", "cpu", False)],
)
async def test_ac_6_8_cuda_presence_follows_only_the_effective_device(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
    requested_device: str,
    effective_device: str,
    has_cuda: bool,
) -> None:
    """Runtime status includes CUDA metadata according to the effective loaded device."""

    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"version-for-{name}")
    fake_runtime.runtime_profiles = [
        SimpleNamespace(
            device=effective_device, device_index=[0], compute_type="resolved", n_mels=80
        )
    ]
    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=3,
        device=requested_device,
        compute_type="requested",
    )

    await engine.prepare()

    status = engine.runtime_status()
    assert status is not None
    assert status.requested_device == requested_device
    assert status.effective_device == effective_device
    assert status.requested_compute_type == "requested"
    assert status.effective_compute_type == "resolved"
    assert (status.cuda is not None) is has_cuda


@pytest.mark.req("SP-LFFASTERWHISPER-002")
def test_ac_7_1_public_engine_and_configuration_are_typed_and_validated(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The public engine exposes required construction and capabilities with immutable mitigation settings."""

    source = _write_model_files(tmp_path / "model")
    mitigation = FasterWhisperSilenceMitigation()
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=7,
        silence_mitigation=mitigation,
    )

    assert isinstance(engine, STTEngineBase)
    assert (
        inspect.signature(FasterWhisperEngine).parameters["model_source"].default is inspect._empty
    )
    assert (
        inspect.signature(FasterWhisperEngine).parameters["cpu_threads"].default is inspect._empty
    )
    assert engine.model_source is source
    assert engine.cpu_threads == 7
    assert engine.capabilities.transport == EngineTransport.IN_PROCESS
    assert engine.capabilities.input_format == EngineInputFormat.PCM_S16LE
    assert engine.capabilities.partial_mode == PartialMode.NONE
    assert engine.capabilities.finalization == frozenset(
        {
            FinalizationMode.ENGINE_SEGMENT,
            FinalizationMode.LOCAL_VAD,
            FinalizationMode.CLIENT_FLUSH,
        }
    )
    assert engine.capabilities.timestamp_detail == TimestampDetail.SEGMENT
    assert engine.capabilities.diarization == DiarizationMode.NONE
    assert engine.capabilities.languages == frozenset({"en", "ja"})
    assert engine.capabilities.auto_language is True
    with pytest.raises(FrozenInstanceError):
        mitigation.vad_filter = False


@pytest.mark.req("SP-LFFASTERWHISPER-002")
@pytest.mark.req("SP-LFFASTERWHISPER-003")
def test_ac_2_1_public_constructor_and_default_profile_are_preserved(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The public constructor exposes the boundary settings and documented default profile."""

    parameters = inspect.signature(FasterWhisperEngine).parameters

    assert list(parameters) == [
        "model_source",
        "cpu_threads",
        "model",
        "language",
        "device",
        "compute_type",
        "beam_size",
        "max_new_tokens",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "initial_prompt",
        "word_timestamps",
        "silence_mitigation",
        "boundary_settings",
        "input_format",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    assert parameters["model_source"].default is inspect.Parameter.empty
    assert parameters["cpu_threads"].default is inspect.Parameter.empty
    assert {
        name: parameter.default
        for name, parameter in parameters.items()
        if name
        not in {
            "model_source",
            "cpu_threads",
        }
    } == {
        "model": "small",
        "language": "en",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 1,
        "max_new_tokens": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "initial_prompt": None,
        "word_timestamps": False,
        "silence_mitigation": FasterWhisperSilenceMitigation(),
        "boundary_settings": UtteranceBoundarySettings(),
        "input_format": EngineInputFormat.PCM_S16LE,
    }
    assert get_type_hints(FasterWhisperEngine.__init__) == {
        "model_source": OfflineModelDirectorySource,
        "cpu_threads": int,
        "model": str,
        "language": str | None,
        "device": str,
        "compute_type": str,
        "beam_size": int,
        "max_new_tokens": int | None,
        "repetition_penalty": float,
        "no_repeat_ngram_size": int,
        "initial_prompt": str | None,
        "word_timestamps": bool,
        "silence_mitigation": FasterWhisperSilenceMitigation,
        "boundary_settings": UtteranceBoundarySettings,
        "input_format": EngineInputFormat,
        "return": type(None),
    }

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
    )

    assert (engine.model, engine.device, engine.compute_type, engine.beam_size) == (
        "small",
        "cpu",
        "int8",
        1,
    )
    assert engine.boundary_settings == UtteranceBoundarySettings()
    assert not hasattr(engine, "silence_threshold_s")
    assert not hasattr(engine, "voice_threshold")


@pytest.mark.req("SP-LFFASTERWHISPER-002")
@pytest.mark.parametrize("invalid_settings", [{"preroll_s": 0}, 0.3])
def test_boundary_settings_requires_core_settings_value_at_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_settings: object,
) -> None:
    """The engine rejects a boundary configuration that is not the core settings value."""

    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loader must not run")),
    )

    with pytest.raises(
        ConfigurationError, match="boundary_settings must be UtteranceBoundarySettings"
    ):
        FasterWhisperEngine(
            model_source=_write_model_files(tmp_path / "model"),
            cpu_threads=1,
            boundary_settings=invalid_settings,  # type: ignore[arg-type]
        )


@pytest.mark.req("SP-LFFASTERWHISPER-005")
async def test_caller_boundary_settings_change_public_utterance_offsets(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The caller's silence and pre-roll settings control public utterance starts and ends."""

    settings = UtteranceBoundarySettings(silence_threshold_s=0.05, preroll_s=0)
    engine, model = await _prepared_engine(tmp_path, fake_runtime, boundary_settings=settings)
    events = [
        event
        async for event in engine.stream(
            _chunks(
                _pcm_chunk(0, duration_s=0.1),
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=0.06),
                _pcm_chunk(700, duration_s=0.01),
            )
        )
    ]

    assert engine.boundary_settings is settings
    assert [type(event) for event in events] == [
        UtteranceStart,
        UtteranceEnd,
        UtteranceStart,
        UtteranceEnd,
    ]
    assert [event.ts_offset_s for event in events] == pytest.approx([0.1, 0.16, 0.17, 0.18])
    assert [len(audio) for audio, _ in model.transcribe_calls] == [1120, 160]
    assert model.transcribe_calls[0][0][0] == pytest.approx(700 / 32768)
    assert model.transcribe_calls[1][0][0] == pytest.approx(700 / 32768)


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "large-v3"),
        ("model", " "),
        ("device", "cuda"),
        ("device", "auto"),
        ("compute_type", "float16"),
        ("compute_type", "auto"),
        ("compute_type", "default"),
        ("beam_size", 2),
        ("beam_size", 5),
    ],
)
def test_ac_2_2_profile_fields_accept_all_shape_valid_values(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    field: str,
    value: object,
) -> None:
    """Callers can retain profile values that meet the documented type and range rules."""

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
        **{field: value},
    )

    assert getattr(engine, field) == value


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("model", None, TypeError, r"model.*str"),
        ("model", "", ValueError, r"model.*non-empty"),
        ("device", 0, TypeError, r"device.*str"),
        ("device", "", ValueError, r"device.*non-empty"),
        ("compute_type", b"int8", TypeError, r"compute_type.*str"),
        ("compute_type", "", ValueError, r"compute_type.*non-empty"),
        ("beam_size", True, TypeError, r"beam_size.*positive integer"),
        ("beam_size", 1.0, TypeError, r"beam_size.*positive integer"),
        ("beam_size", 0, ValueError, r"beam_size.*positive integer"),
        ("beam_size", -1, ValueError, r"beam_size.*positive integer"),
    ],
)
def test_ac_2_2_invalid_profile_fields_fail_before_runtime_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid profile values fail with the documented error before the optional runtime loads."""

    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loader must not run")),
    )

    with pytest.raises(ConfigurationError, match=message) as captured:
        FasterWhisperEngine(
            model_source=_write_model_files(tmp_path / "model"),
            cpu_threads=1,
            **{field: value},
        )
    assert isinstance(captured.value.__cause__, error_type)


@pytest.mark.req("SP-LFFASTERWHISPER-002")
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cpu_threads": 0}, "cpu_threads.*positive integer"),
        ({"initial_prompt": ""}, "initial_prompt.*non-empty"),
        ({"word_timestamps": 1}, "word_timestamps.*bool"),
    ],
)
def test_ac_7_1_constructor_rejects_invalid_non_profile_values(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    overrides: dict[str, object],
    message: str,
) -> None:
    """The public constructor rejects invalid non-profile configuration values."""

    arguments: dict[str, object] = {
        "model_source": _write_model_files(tmp_path / "model"),
        "cpu_threads": 1,
    }
    arguments.update(overrides)

    with pytest.raises(ConfigurationError, match=message):
        FasterWhisperEngine(**arguments)  # type: ignore[arg-type]


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "device", "compute_type"),
    [
        ("large-v3", "cuda", "float16"),
        ("custom-label", "auto", "auto"),
        ("another-label", "cpu", "default"),
    ],
)
async def test_ac_2_3_profile_reaches_the_single_offline_model_load_unchanged(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    model: str,
    device: str,
    compute_type: str,
) -> None:
    """A selected profile reaches one offline model load without substitution."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=11,
        model=model,
        device=device,
        compute_type=compute_type,
    )

    await engine.prepare()

    loaded = fake_runtime.instances[0]
    assert loaded.model_path == source.directory_path
    assert loaded.load_kwargs == {
        "device": device,
        "device_index": 0,
        "compute_type": compute_type,
        "cpu_threads": 11,
        "num_workers": 1,
        "local_files_only": True,
    }
    assert len(fake_runtime.instances) == 1


@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.asyncio
async def test_ac_2_3_model_label_never_changes_the_weights_directory(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The model label does not change the caller-supplied weights directory."""

    source = _write_model_files(tmp_path / "model")

    for model in ("small", "large-v3"):
        engine = FasterWhisperEngine(
            model_source=source,
            cpu_threads=1,
            model=model,
        )
        await engine.prepare()

    assert [loaded.model_path for loaded in fake_runtime.instances] == [
        source.directory_path,
        source.directory_path,
    ]


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.asyncio
async def test_ac_2_4_beam_size_reaches_warm_up_and_every_utterance_unchanged(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The selected beam size reaches warm-up and each utterance without substitution."""

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
        beam_size=5,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )

    await engine.prepare()
    model = fake_runtime.instances[0]
    model.responses.extend(
        [
            [SimpleNamespace(text="first", start=0.0)],
            [SimpleNamespace(text="second", start=0.0)],
        ]
    )
    events = [
        event
        async for event in engine.stream(
            _chunks(
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=0.1),
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=0.1),
            )
        )
    ]

    assert len(model.transcribe_calls) == 3
    assert [kwargs["beam_size"] for _, kwargs in model.transcribe_calls] == [5, 5, 5]
    assert [event.text for event in events if isinstance(event, Final)] == ["first", "second"]


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.parametrize(
    ("failure_site", "upstream", "error_type"),
    [
        ("configuration", ValueError("cpu_threads must be a positive integer"), ConfigurationError),
        (
            "missing runtime",
            ModuleNotFoundError("blocked faster_whisper", name="faster_whisper"),
            ModuleNotInstalledError,
        ),
        (
            "native import",
            OSError("libcudnn.so.9: cannot open shared object file"),
            EnvironmentUnavailableError,
        ),
        ("native import", RuntimeError("native module initialization failed"), InferenceError),
        ("model construction", RuntimeError("load failed"), InferenceError),
        (
            "model construction",
            OSError("libgomp.so.1: cannot open shared object file: No such file or directory"),
            EnvironmentUnavailableError,
        ),
        ("model construction", ValueError("unsupported compute type"), ConfigurationError),
        (
            "model construction",
            RuntimeError("CUDA failed with error no CUDA-capable device is detected"),
            EnvironmentUnavailableError,
        ),
        (
            "model construction",
            RuntimeError(
                "CUDA failed with error CUDA driver version is insufficient for CUDA runtime version"
            ),
            EnvironmentUnavailableError,
        ),
        (
            "warm-up call",
            RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"),
            EnvironmentUnavailableError,
        ),
        (
            "warm-up call",
            OSError("libcudnn.so.9: cannot open shared object file: No such file or directory"),
            EnvironmentUnavailableError,
        ),
        ("warm-up call", RuntimeError("warm-up failed"), InferenceError),
        ("warm-up drain", RuntimeError("warm-up drain failed"), InferenceError),
        ("utterance call", RuntimeError("decode failed"), InferenceError),
        ("utterance call", RuntimeError("CUDA failed with error out of memory"), InferenceError),
        ("utterance drain", RuntimeError("decode drain failed"), InferenceError),
    ],
)
async def test_native_and_upstream_failures_use_core_errors_with_the_original_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    upstream: Exception,
    error_type: type[Exception],
) -> None:
    """Configuration, missing runtime, loading, warm-up, and decoding errors retain their cause without fallback."""

    source = _write_model_files(tmp_path / "model")
    if failure_site == "configuration":
        with pytest.raises(ConfigurationError) as captured:
            FasterWhisperEngine(model_source=source, cpu_threads=0)
        assert isinstance(captured.value.__cause__, ValueError)
        return
    if failure_site in {"missing runtime", "native import"}:
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise upstream
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        with pytest.raises(error_type) as captured:
            FasterWhisperEngine(model_source=source, cpu_threads=1)
        assert captured.value.__cause__ is upstream
        return

    FaultInjectingWhisperModel.configure(failure_site=failure_site, failure=upstream)
    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model", lambda: FaultInjectingWhisperModel
    )
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=3,
        model="large-v3",
        device="cuda",
        compute_type="float16",
        beam_size=5,
    )
    preparing = not failure_site.startswith("utterance")
    if not preparing:
        await engine.prepare()
    with pytest.raises(error_type) as captured:
        if preparing:
            await engine.prepare()
        else:
            async for _ in engine.stream(_chunks(_pcm_chunk(700, duration_s=0.01))):
                pass
    assert captured.value.__cause__ is upstream
    assert engine.health().state is (EngineState.UNPREPARED if preparing else EngineState.READY)
    if preparing:
        with pytest.raises(EngineStateError):
            engine.runtime_status()
    else:
        assert engine.runtime_status() is not None
    assert len(FaultInjectingWhisperModel.construction_calls) == 1
    _, kwargs = FaultInjectingWhisperModel.construction_calls[0]
    assert kwargs["device"] == "cuda" and kwargs["compute_type"] == "float16"
    assert len(FaultInjectingWhisperModel.transcribe_calls) == (
        0 if failure_site == "model construction" else 1 if preparing else 2
    )
    assert all(
        kwargs["beam_size"] == 5 for _, kwargs in FaultInjectingWhisperModel.transcribe_calls
    )
    await engine.aclose()


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vad_filter": 1}, "vad_filter.*bool"),
        ({"vad_threshold": -0.01}, "vad_threshold.*0.0.*1.0"),
        ({"vad_threshold": 1.01}, "vad_threshold.*0.0.*1.0"),
        ({"vad_min_silence_duration_ms": -1}, "vad_min_silence_duration_ms.*0 or greater"),
        ({"no_speech_threshold": 1.01}, "no_speech_threshold.*0.0.*1.0"),
        ({"condition_on_previous_text": 0}, "condition_on_previous_text.*bool"),
        (
            {"hallucination_silence_threshold_s": 0},
            "hallucination_silence_threshold_s.*greater than 0",
        ),
        (
            {"hallucination_silence_threshold_s": float("inf")},
            "hallucination_silence_threshold_s.*finite.*greater than 0",
        ),
    ],
)
def test_ac_7_1_silence_mitigation_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Silence mitigation controls reject values outside their documented types and ranges."""

    with pytest.raises(ConfigurationError, match=message):
        FasterWhisperSilenceMitigation(**kwargs)  # type: ignore[arg-type]


@pytest.mark.req("SP-LFFASTERWHISPER-004")
def test_ac_7_2_model_source_is_immutable_and_requires_an_absolute_directory() -> None:
    """The offline model source requires an absolute directory and retains its identity."""

    with pytest.raises(ValueError, match="directory_path.*absolute"):
        OfflineModelDirectorySource(directory_path="relative/model")

    source = OfflineModelDirectorySource(
        directory_path=_absolute_model_path("faster-whisper-small")
    )
    with pytest.raises(FrozenInstanceError):
        source.directory_path = _absolute_model_path("replacement")


@pytest.mark.req("SP-LFFASTERWHISPER-004")
def test_ac_7_2_construction_rejects_a_missing_tokenizer_before_runtime_load(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Construction rejects a missing tokenizer before loading the runtime model."""

    source = _write_model_files(tmp_path / "model")
    missing = Path(source.directory_path) / "tokenizer.json"
    missing.unlink()
    with pytest.raises(ConfigurationError, match="tokenizer.json"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)

    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.asyncio
async def test_ac_5_1_prepare_accepts_vocabulary_json_in_place_of_vocabulary_txt(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Preparation accepts a vocabulary JSON file as the offline vocabulary asset."""

    source = _write_model_files(tmp_path / "model")
    directory = Path(source.directory_path)
    (directory / "vocabulary.txt").unlink()
    (directory / "vocabulary.json").write_text(json.dumps(["token", ""]))
    engine = FasterWhisperEngine(model_source=source, cpu_threads=1)

    await engine.prepare()

    assert len(fake_runtime.instances) == 1
    assert fake_runtime.instances[0].model_path == str(directory)
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTERWHISPER-004")
def test_ac_5_2_construction_names_both_vocabulary_files_when_neither_exists(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Construction identifies both supported vocabulary filenames when neither asset exists."""

    source = _write_model_files(tmp_path / "model")
    (Path(source.directory_path) / "vocabulary.txt").unlink()
    with pytest.raises(ConfigurationError, match=r"vocabulary\.json or vocabulary\.txt"):
        FasterWhisperEngine(model_source=source, cpu_threads=1)

    assert fake_runtime.instances == []


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.asyncio
async def test_ac_7_4_and_ac_7_5_prepare_forwards_offline_load_and_decode_profile(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Preparation loads only the selected offline directory and forwards the decoding configuration."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=11,
        language="ja",
        initial_prompt="listen flow",
    )

    await engine.prepare()

    model = fake_runtime.instances[0]
    assert model.model_path == source.directory_path
    assert model.load_kwargs == {
        "device": "cpu",
        "device_index": 0,
        "compute_type": "int8",
        "cpu_threads": 11,
        "num_workers": 1,
        "local_files_only": True,
    }
    assert len(model.transcribe_calls) == 1
    _, kwargs = model.transcribe_calls[0]
    assert kwargs == {
        "language": "ja",
        "beam_size": 1,
        "initial_prompt": "listen flow",
        "word_timestamps": False,
        "vad_filter": True,
        "vad_parameters": {"threshold": 0.65, "min_silence_duration_ms": 500},
        "no_speech_threshold": 0.75,
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 1.0,
    }
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.asyncio
async def test_ac_7_6_bias_evidence_changes_only_after_a_successful_decode_drain(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt application evidence becomes true only after the prompted decode drains successfully."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        initial_prompt="Example Prompt",
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    control = _ControlledRuntime(
        loop=asyncio.get_running_loop(),
        responses=[[SimpleNamespace(text=" hello ", start=0.01)]],
        blocked_calls={0},
    )
    monkeypatch.setattr(model, "transcribe", control.transcribe)
    iterator = engine.stream(
        _chunks(
            _pcm_chunk(700, duration_s=0.01),
            _pcm_chunk(0, duration_s=0.1),
        )
    )

    assert await anext(iterator) == UtteranceStart(utterance_id=1, ts_offset_s=0.0)
    await control.started[0].wait()
    assert engine.health().bias_application == BiasApplicationSnapshot(
        session_id=1,
        kind="prompt",
        prompt="Example Prompt",
        applied_value="Example Prompt",
        applied=False,
        evidence_method="verified_apply_path",
        evidence_detail=("faster_whisper.WhisperModel.transcribe(..., initial_prompt=...)"),
    )

    control.releases[0].set()
    remaining = [event async for event in iterator]

    assert isinstance(remaining[-1], UtteranceEnd)
    assert engine.health().bias_application is not None
    assert engine.health().bias_application.applied is True
    assert control.kwargs[0]["initial_prompt"] == "Example Prompt"


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.asyncio
async def test_ac_7_6_bias_evidence_stays_false_on_decode_failure_and_resets_next_session(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Prompt application evidence stays false after decode failure and starts fresh in the next session."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        initial_prompt="prompt",
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    model.responses.append(RuntimeError("decode failed"))

    with pytest.raises(RuntimeError, match="decode failed"):
        async for _ in engine.stream(
            _chunks(
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=0.1),
            )
        ):
            pass

    failed = engine.health().bias_application
    assert failed is not None
    assert failed.session_id == 1
    assert failed.applied is False

    model.responses.append([])
    events = [event async for event in engine.stream(_chunks(_pcm_chunk(700, duration_s=0.01)))]

    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.01),
    ]
    current = engine.health().bias_application
    assert current is not None
    assert current.session_id == 2
    assert current.applied is True


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.asyncio
async def test_ac_10_13_bias_evidence_is_sticky_after_a_later_decode_failure(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later decode failure does not erase prompt application evidence earned in the session."""

    expected = RuntimeError("later decode failed")
    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        initial_prompt="sticky prompt",
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    control = _ControlledRuntime(
        loop=asyncio.get_running_loop(),
        responses=[
            [SimpleNamespace(text="first", start=0.0)],
            expected,
        ],
        blocked_calls={0, 1},
    )
    monkeypatch.setattr(model, "transcribe", control.transcribe)
    iterator = engine.stream(
        _chunks(
            _pcm_chunk(700, duration_s=0.01),
            _pcm_chunk(0, duration_s=0.06),
            _pcm_chunk(800, duration_s=0.01),
            _pcm_chunk(0, duration_s=0.06),
        )
    )
    try:
        # Pause event consumption while the later decode completes.
        await anext(iterator)
        await control.started[0].wait()
        control.releases[0].set()
        await control.started[1].wait()
        applied = engine.health().bias_application
        assert control.drained == [True, False]
        assert applied is not None
        assert applied.session_id == 1
        assert applied.applied is True

        control.releases[1].set()
        await control.finished[1].wait()
        # Wait for the pipeline to observe the failure before consuming again.
        while engine.health().decoding_utterance_id is not None:
            await _run_ready_callbacks()

        with pytest.raises(RuntimeError) as captured:
            async for _ in iterator:
                pass

        assert captured.value.__cause__ is expected
        assert engine.health().bias_application == applied
    finally:
        control.releases[0].set()
        control.releases[1].set()
        await iterator.aclose()


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.asyncio
async def test_ac_10_13_session_prompt_is_fixed_at_first_consumption(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """A stream retains the prompt selected when consumption begins for its decode and health evidence."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        initial_prompt="session prompt",
    )
    model.responses.append([])
    first_pull = asyncio.Event()
    allow_pcm = asyncio.Event()

    async def paused_input() -> AsyncIterator[bytes]:
        first_pull.set()
        await allow_pcm.wait()
        yield _pcm_chunk(700, duration_s=0.01)

    consumer = asyncio.create_task(
        _collect_events(engine.stream(paused_input())),
        name="test-ac-10-13-consumer",
    )
    await first_pull.wait()
    snapshot = engine.health().bias_application
    assert snapshot is not None
    assert snapshot.prompt == "session prompt"
    engine.initial_prompt = "later engine value"
    allow_pcm.set()

    await consumer

    assert model.transcribe_calls[0][1]["initial_prompt"] == "session prompt"
    applied = engine.health().bias_application
    assert applied is not None
    assert applied.applied_value == "session prompt"
    assert applied.applied is True


@pytest.mark.req("SP-LFFASTERWHISPER-006")
@pytest.mark.asyncio
async def test_ac_10_3_decode_does_not_stop_later_pcm_intake(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCM intake continues during decoding while later utterances decode serially with ordered events."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    control = _ControlledRuntime(
        loop=asyncio.get_running_loop(),
        responses=[
            [SimpleNamespace(text="first", start=0.0)],
            [SimpleNamespace(text="second", start=0.0)],
        ],
        blocked_calls={0, 1},
    )
    monkeypatch.setattr(model, "transcribe", control.transcribe)
    chunks = [
        _pcm_chunk(700, duration_s=0.01),
        _pcm_chunk(0, duration_s=0.06),
        _pcm_chunk(800, duration_s=0.01),
        _pcm_chunk(0, duration_s=0.06),
    ]
    source = _GatedChunks(
        chunks,
        gate=control.started[0],
        gate_before_index=2,
    )
    consumer = asyncio.create_task(
        _collect_events(engine.stream(source)),
        name="test-ac-10-3-consumer",
    )

    await control.started[0].wait()
    try:
        await _run_ready_callbacks()
        assert control.drained[0] is False
        assert source.exhaustion_requested.is_set()
        assert source.yielded == chunks
        live = engine.health()
        assert live.processed_pcm_chunks == 4
        assert live.decoding_utterance_id == 1
        assert live.pending_decode_utterances == 1
        last_progress = live.last_progress_monotonic_s
        await _run_ready_callbacks()
        assert engine.health().last_progress_monotonic_s == last_progress

        control.releases[0].set()
        await control.started[1].wait()
        second_decode = engine.health()
        assert second_decode.decoding_utterance_id == 2
        assert second_decode.pending_decode_utterances == 0
        control.releases[1].set()
        await consumer
    finally:
        control.releases[0].set()
        control.releases[1].set()
        if not consumer.done():
            await consumer

    events = consumer.result()

    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        Partial(utterance_id=1, ts_offset_s=0.0, text="first"),
        Final(utterance_id=1, ts_offset_s=0.0, text="first"),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.06)),
        UtteranceStart(utterance_id=2, ts_offset_s=pytest.approx(0.07)),
        Partial(utterance_id=2, ts_offset_s=pytest.approx(0.07), text="second"),
        Final(utterance_id=2, ts_offset_s=pytest.approx(0.07), text="second"),
        UtteranceEnd(utterance_id=2, ts_offset_s=pytest.approx(0.13)),
    ]
    assert control.audio[0] == pytest.approx(
        [700 / 32768] * round(16_000 * 0.01) + [0.0] * round(16_000 * 0.06)
    )
    assert control.audio[1] == pytest.approx(
        [800 / 32768] * round(16_000 * 0.01) + [0.0] * round(16_000 * 0.06)
    )
    assert control.drained == [True, True]
    assert control.max_active_calls == 1


@pytest.mark.req("SP-LFFASTERWHISPER-007")
@pytest.mark.asyncio
async def test_ac_10_8_decode_failure_discards_waiting_work_and_cleans_up(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode failure discards waiting work, clears live state, and permits engine reuse."""

    expected = RuntimeError("controlled decode failure")
    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    control = _ControlledRuntime(
        loop=asyncio.get_running_loop(),
        responses=[expected, ()],
        blocked_calls={0},
    )
    monkeypatch.setattr(model, "transcribe", control.transcribe)
    chunks = [
        _pcm_chunk(700, duration_s=0.01),
        _pcm_chunk(0, duration_s=0.06),
        _pcm_chunk(800, duration_s=0.01),
        _pcm_chunk(0, duration_s=0.06),
    ]
    source = _GatedChunks(
        chunks,
        gate=control.started[0],
        gate_before_index=2,
    )
    delivered: list[object] = []
    consumer = asyncio.create_task(
        _collect_events_into(engine.stream(source), delivered),
        name="test-ac-10-8-consumer",
    )
    terminal: BaseException | None = None

    await control.started[0].wait()
    try:
        await _run_ready_callbacks()
        assert source.exhaustion_requested.is_set()
        assert engine.health().pending_decode_utterances == 1
    finally:
        control.releases[0].set()
        try:
            await consumer
        except RuntimeError as error:
            terminal = error

    assert isinstance(terminal, InferenceError)
    assert terminal.__cause__ is expected
    assert len(control.audio) == 1
    cleaned = engine.health()
    assert cleaned.processed_pcm_chunks == 4
    assert cleaned.emitted_events == len(delivered)
    assert cleaned.decoding_utterance_id is None
    assert cleaned.pending_decode_utterances == 0
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("listenflow-utterance-pipeline-")
    ]

    reused = [event async for event in engine.stream(_chunks(_pcm_chunk(900, duration_s=0.01)))]
    assert reused == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.01),
    ]


@pytest.mark.req("SP-LFFASTERWHISPER-007")
@pytest.mark.asyncio
async def test_ac_10_9_input_failure_waits_for_running_decode_cleanup(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input failure keeps the stream active until its running decode finishes cleanup."""

    expected = RuntimeError("controlled input failure")
    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        initial_prompt="sticky prompt",
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    control = _ControlledRuntime(
        loop=asyncio.get_running_loop(),
        responses=[[SimpleNamespace(text="discarded", start=0.0)], ()],
        blocked_calls={0},
    )
    monkeypatch.setattr(model, "transcribe", control.transcribe)
    source = _GatedChunks(
        [
            _pcm_chunk(700, duration_s=0.01),
            _pcm_chunk(0, duration_s=0.06),
        ],
        gate=control.started[0],
        gate_before_index=2,
        failure=expected,
    )
    delivered: list[object] = []
    consumer = asyncio.create_task(
        _collect_events_into(engine.stream(source), delivered),
        name="test-ac-10-9-consumer",
    )
    terminal: BaseException | None = None

    await control.started[0].wait()
    try:
        await _run_ready_callbacks()
        assert source.failure_raised.is_set()
        assert not consumer.done()
        assert engine.health().decoding_utterance_id == 1
        second = engine.stream(_chunks())
        with pytest.raises(
            RuntimeError,
            match="engine already has an active stream",
        ):
            await anext(second)
    finally:
        control.releases[0].set()
        try:
            await consumer
        except RuntimeError as error:
            terminal = error

    assert terminal is expected
    assert len(control.audio) == 1
    assert control.drained[0] is True
    cleaned = engine.health()
    assert cleaned.processed_pcm_chunks == 2
    assert cleaned.emitted_events == len(delivered)
    assert cleaned.decoding_utterance_id is None
    assert cleaned.pending_decode_utterances == 0
    bias = cleaned.bias_application
    assert bias is not None
    assert bias.applied is True

    reused = [event async for event in engine.stream(_chunks(_pcm_chunk(900, duration_s=0.01)))]
    assert reused[-1] == UtteranceEnd(utterance_id=1, ts_offset_s=0.01)


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-007")
@pytest.mark.parametrize("termination", ["caller cancellation", "iterator close", "engine close"])
async def test_cancellation_and_closure_wait_for_in_progress_transcription(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    """Cancellation and either close wait for native transcription before releasing the stream or model."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        initial_prompt="cleanup prompt",
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    control = _ControlledRuntime(
        loop=asyncio.get_running_loop(), responses=[(), ()], blocked_calls={0}
    )
    monkeypatch.setattr(model, "transcribe", control.transcribe)
    released = []
    model.model.unload_model = lambda: released.append(control.finished[0].is_set())
    iterator = engine.stream(
        _chunks(_pcm_chunk(700, duration_s=0.01), _pcm_chunk(0, duration_s=0.06))
    )
    assert await anext(iterator) == UtteranceStart(utterance_id=1, ts_offset_s=0.0)
    await control.started[0].wait()
    read = asyncio.create_task(anext(iterator))
    await _run_ready_callbacks()
    if termination == "caller cancellation":
        read.cancel()
        stopping = read
    elif termination == "iterator close":
        stopping = asyncio.create_task(iterator.aclose())
    else:
        stopping = asyncio.create_task(engine.aclose())
    try:
        await _run_ready_callbacks()
        assert not stopping.done()
        assert not control.finished[0].is_set()
        assert released == []
        assert engine.health().busy is True
        assert engine.health().decoding_utterance_id == 1
        if termination == "engine close":
            with pytest.raises(EngineClosedError):
                engine.runtime_status()
        else:
            assert engine.runtime_status() is not None
        with pytest.raises(
            EngineClosedError if termination == "engine close" else EngineStateError
        ):
            await anext(engine.stream(_chunks()))
        # Repeated cancellation must not abandon native work or release the stream early.
        if termination == "caller cancellation":
            read.cancel()
            await _run_ready_callbacks()
            assert not read.done()
    finally:
        control.releases[0].set()
        if termination == "caller cancellation":
            with pytest.raises(asyncio.CancelledError):
                await stopping
        else:
            await stopping
        if stopping is not read:
            with pytest.raises(
                EngineClosedError if termination == "engine close" else asyncio.CancelledError
            ):
                await read
        await control.finished[0].wait()
    assert control.drained[0] is True
    health = engine.health()
    assert health.busy is False
    assert health.decoding_utterance_id is None and health.pending_decode_utterances == 0
    if termination == "engine close":
        assert released == [True]
        assert health.state is EngineState.CLOSED
        with pytest.raises(EngineClosedError):
            engine.runtime_status()
        await engine.aclose()
        assert released == [True]
    else:
        assert health.state is EngineState.READY
        reused = [event async for event in engine.stream(_chunks(_pcm_chunk(900, duration_s=0.01)))]
        assert reused[-1] == UtteranceEnd(utterance_id=1, ts_offset_s=0.01)
        await engine.aclose()


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_format",
    [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32],
)
async def test_ac_7_7_multiple_segments_and_utterances_preserve_public_event_semantics(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
) -> None:
    """Multiple segments and utterances emit ordered public transcript events with source offsets."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
        input_format=input_format,
    )
    model.responses.extend(
        [
            [
                SimpleNamespace(text=" hello ", start=0.01),
                SimpleNamespace(text=" world ", start=0.02),
            ],
            [SimpleNamespace(text="again", start=0.0)],
        ]
    )

    events = [
        event
        async for event in engine.stream(
            _chunks(
                _format_chunk(input_format, 700, duration_s=0.01),
                _format_chunk(input_format, 0, duration_s=0.1),
                _format_chunk(input_format, 800, duration_s=0.01),
                _format_chunk(input_format, 0, duration_s=0.1),
            )
        )
    ]

    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        Partial(utterance_id=1, ts_offset_s=0.01, text="hello"),
        Final(utterance_id=1, ts_offset_s=0.01, text="hello"),
        Partial(utterance_id=1, ts_offset_s=0.02, text="hello world"),
        Final(utterance_id=1, ts_offset_s=0.02, text="world"),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.06)),
        UtteranceStart(utterance_id=2, ts_offset_s=pytest.approx(0.11)),
        Partial(utterance_id=2, ts_offset_s=pytest.approx(0.11), text="again"),
        Final(utterance_id=2, ts_offset_s=pytest.approx(0.11), text="again"),
        UtteranceEnd(utterance_id=2, ts_offset_s=pytest.approx(0.17)),
    ]
    finals = [event for event in events if isinstance(event, Final)]
    assert [event.text for event in finals] == ["hello", "world", "again"]


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
async def test_ac_10_5_segment_conversion_is_materialized_with_source_offsets(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Non-empty decoder segments produce cumulative and final events at their source offsets."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    model.responses.append(
        [
            SimpleNamespace(text=" alpha ", start=0.25),
            SimpleNamespace(text="   ", start=0.4),
            SimpleNamespace(text=" beta ", start=None),
        ]
    )

    events = [
        event
        async for event in engine.stream(
            _chunks(
                _pcm_chunk(0, duration_s=0.5),
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=0.06),
            )
        )
    ]

    # The decoded utterance starts with the core's 300 ms pre-roll at 0.2 s.
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=pytest.approx(0.2)),
        Partial(utterance_id=1, ts_offset_s=pytest.approx(0.45), text="alpha"),
        Final(utterance_id=1, ts_offset_s=pytest.approx(0.45), text="alpha"),
        Partial(utterance_id=1, ts_offset_s=pytest.approx(0.2), text="alpha beta"),
        Final(utterance_id=1, ts_offset_s=pytest.approx(0.2), text="beta"),
        UtteranceEnd(utterance_id=1, ts_offset_s=pytest.approx(0.56)),
    ]
    assert len(model.transcribe_calls) == 1


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_format",
    [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32],
)
async def test_ac_7_8_digital_silence_and_empty_transcript_have_distinct_boundaries(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
) -> None:
    """Digital silence emits no events while speech with no transcript emits only boundaries."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
        input_format=input_format,
    )

    silence_events = [
        event
        async for event in engine.stream(_chunks(_format_chunk(input_format, 0, duration_s=0.2)))
    ]

    assert silence_events == []
    assert engine.health().bias_application is not None
    assert engine.health().bias_application.applied is False
    assert model.transcribe_calls == []

    model.responses.append([])
    empty_events = [
        event
        async for event in engine.stream(_chunks(_format_chunk(input_format, 700, duration_s=0.01)))
    ]

    assert empty_events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.01),
    ]
    assert not any(isinstance(event, (Partial, Final)) for event in empty_events)


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_format",
    [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32],
)
async def test_ac_7_9_silence_boundary_and_input_flush_end_after_decode(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
) -> None:
    """Silence boundaries and input exhaustion emit the utterance end after transcript decoding."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
        input_format=input_format,
    )
    model.responses.extend(
        [
            [SimpleNamespace(text="boundary", start=0.0)],
            [SimpleNamespace(text="flush", start=0.0)],
        ]
    )

    boundary_events = [
        event
        async for event in engine.stream(
            _chunks(
                _format_chunk(input_format, 700, duration_s=0.01),
                _format_chunk(input_format, 0, duration_s=0.1),
            )
        )
    ]
    flush_events = [
        event
        async for event in engine.stream(
            _chunks(_format_chunk(input_format, 700, duration_s=0.025))
        )
    ]

    assert isinstance(boundary_events[-2], Final)
    assert boundary_events[-1] == UtteranceEnd(
        utterance_id=1,
        ts_offset_s=pytest.approx(0.06),
    )
    assert isinstance(flush_events[-2], Final)
    assert flush_events[-1] == UtteranceEnd(
        utterance_id=1,
        ts_offset_s=pytest.approx(0.025),
    )


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
async def test_ac_7_9_odd_byte_pcm_is_rejected(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """A PCM chunk with an incomplete signed sample is rejected."""

    engine, _ = await _prepared_engine(tmp_path, fake_runtime)

    with pytest.raises(ConfigurationError, match=r"pcm_s16le.*multiple of 2 bytes.*whole samples"):
        async for _ in engine.stream(_chunks(b"\x00")):
            pass


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.asyncio
async def test_ac_7_10_health_preserves_existing_counters_and_tracks_event_emission(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Health counts public PCM chunks and delivered events across sequential streams."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=1,
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.05),
    )
    before = engine.health()
    assert before.state is EngineState.UNPREPARED
    assert before.processed_pcm_chunks == 0
    assert before.emitted_events == 0
    assert before.last_progress_monotonic_s is None
    assert before.bias_application is None
    assert before.decoding_utterance_id is None
    assert before.pending_decode_utterances == 0

    await engine.prepare()
    prepared_status = engine.runtime_status()
    assert prepared_status is not None
    assert engine.runtime_status() == prepared_status
    assert engine.health().processed_pcm_chunks == 0
    assert engine.health().emitted_events == 0
    fake_runtime.instances[0].transcribe_calls.clear()
    fake_runtime.instances[0].responses.append([SimpleNamespace(text="hello", start=0.0)])
    events = [
        event
        async for event in engine.stream(
            _chunks(
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=0.1),
            )
        )
    ]

    after = engine.health()
    assert after.state is EngineState.READY
    assert after.processed_pcm_chunks == 2
    assert after.emitted_events == len(events) == 4
    assert after.last_progress_monotonic_s is not None
    assert after.decoding_utterance_id is None
    assert after.pending_decode_utterances == 0

    fake_runtime.instances[0].responses.append([])
    second_events = [
        event async for event in engine.stream(_chunks(_pcm_chunk(800, duration_s=0.01)))
    ]
    lifetime = engine.health()
    assert lifetime.processed_pcm_chunks == 3
    assert lifetime.emitted_events == len(events) + len(second_events) == 6
    assert lifetime.last_progress_monotonic_s != after.last_progress_monotonic_s
    assert lifetime.decoding_utterance_id is None
    assert lifetime.pending_decode_utterances == 0


@pytest.mark.req("SP-LFFASTERWHISPER-007")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_format",
    [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32],
)
async def test_ac_7_11_second_concurrent_stream_fails_without_stopping_first(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
) -> None:
    """A second active stream is rejected without stopping the first stream."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        input_format=input_format,
    )
    model.responses.append([SimpleNamespace(text="first", start=0.0)])
    voice = _format_chunk(input_format, 700, duration_s=0.01)
    first = engine.stream(_chunks(voice))

    assert await anext(first) == UtteranceStart(utterance_id=1, ts_offset_s=0.0)
    second = engine.stream(_chunks(voice))
    with pytest.raises(RuntimeError, match="engine.*active stream"):
        await anext(second)

    remaining = [event async for event in first]

    assert isinstance(remaining[-2], Final)
    assert isinstance(remaining[-1], UtteranceEnd)


@pytest.mark.req("SP-LFFASTERWHISPER-002")
def test_ac_7_12_core_import_succeeds_without_faster_whisper_until_engine_construction() -> None:
    """Core types remain importable without faster-whisper and engine construction explains the missing runtime."""

    code = r"""
import builtins
from pathlib import Path
from tempfile import TemporaryDirectory

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "faster_whisper" or name.startswith("faster_whisper."):
        raise ModuleNotFoundError("blocked faster_whisper", name="faster_whisper")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import

import listenflow
from listenflow import Final, ModuleNotInstalledError, OfflineModelDirectorySource

assert Final(utterance_id=1, ts_offset_s=0.0, text="x").text == "x"
from listenflow.faster_whisper import FasterWhisperEngine

with TemporaryDirectory() as temp_dir:
    directory = Path(temp_dir)
    (directory / "config.json").write_text('{"lang_ids": [100]}')
    (directory / "tokenizer.json").write_text(
        '{"added_tokens": [{"id": 100, "content": "<|en|>"}]}'
    )
    (directory / "vocabulary.txt").write_text("token\n\n")
    source = OfflineModelDirectorySource(directory_path=str(directory))
    try:
        FasterWhisperEngine(model_source=source, cpu_threads=1)
    except ModuleNotInstalledError as exc:
        assert isinstance(exc.__cause__, ModuleNotFoundError)
        assert "pip install listenflow-faster-whisper" in str(exc)
    else:
        raise AssertionError("FasterWhisperEngine should require its runtime dependency")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-LFFASTERWHISPER-002")
def test_ac_7_13_public_exports_distinguish_direct_engine_configuration() -> None:
    """The engine module exports its own public configuration without adding it to core exports."""

    assert listenflow.faster_whisper.FasterWhisperEngine is FasterWhisperEngine
    assert (
        listenflow.faster_whisper.FasterWhisperSilenceMitigation is FasterWhisperSilenceMitigation
    )
    assert listenflow.OfflineModelDirectorySource is OfflineModelDirectorySource
    assert listenflow.BiasApplicationSnapshot is BiasApplicationSnapshot
    assert "FasterWhisperEngine" in listenflow.faster_whisper.__all__
    assert "FasterWhisperSilenceMitigation" in listenflow.faster_whisper.__all__
    assert "FasterWhisperEngine" not in listenflow.__all__
    assert "FasterWhisperSilenceMitigation" not in listenflow.__all__


@pytest.mark.req("SP-LFFASTERWHISPER-005")
def test_ac_11_1_faster_whisper_capability_tracks_constructor_format(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Capabilities report the fixed PCM format selected at construction."""

    source = _write_model_files(tmp_path / "model")

    default_engine = FasterWhisperEngine(model_source=source, cpu_threads=1)
    float_engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=1,
        input_format=EngineInputFormat.PCM_F32,
    )

    assert default_engine.capabilities.input_format is EngineInputFormat.PCM_S16LE
    assert float_engine.capabilities.input_format is EngineInputFormat.PCM_F32
    with pytest.raises(AttributeError):
        float_engine.input_format = EngineInputFormat.PCM_S16LE


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.parametrize(
    "invalid_format",
    [EngineInputFormat.WAV_PCM_S16LE, EngineInputFormat.BASE64_PCM16, "pcm_f32"],
)
def test_ac_11_2_invalid_format_precedes_optional_runtime_load(
    monkeypatch: pytest.MonkeyPatch,
    invalid_format: object,
) -> None:
    """Unsupported input formats fail before the optional runtime loads."""

    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loader must not run")),
    )

    with pytest.raises(
        ValueError,
        match=r"FasterWhisperEngine.*input_format.*pcm_s16le.*pcm_f32",
    ):
        FasterWhisperEngine(
            model_source=OfflineModelDirectorySource(directory_path=_absolute_model_path("model")),
            cpu_threads=1,
            input_format=invalid_format,  # type: ignore[arg-type]
        )


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.parametrize(
    "invalid_threshold",
    [True, False, 0, 1, 500, "0.1", None, float("nan"), float("inf"), -float("inf"), -0.1, 1.1],
)
def test_ac_11_6_core_settings_reject_invalid_threshold_before_runtime_load(
    monkeypatch: pytest.MonkeyPatch,
    invalid_threshold: object,
) -> None:
    """Core settings reject invalid voice thresholds before the optional runtime loads."""

    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loader must not run")),
    )

    with pytest.raises(
        ValueError,
        match=r"voice_threshold.*normalized full-scale.*\[0\.0, 1\.0\].*500.*0\.01526",
    ):
        FasterWhisperEngine(
            model_source=OfflineModelDirectorySource(directory_path=_absolute_model_path("model")),
            cpu_threads=1,
            boundary_settings=UtteranceBoundarySettings(
                voice_threshold=invalid_threshold,  # type: ignore[arg-type]
            ),
        )


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
async def test_ac_11_3_float32_decoder_input_is_not_clipped_scaled_or_requantized(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Float32 PCM reaches the decoder without clipping, scaling, or requantization."""

    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        input_format=EngineInputFormat.PCM_F32,
        boundary_settings=UtteranceBoundarySettings(voice_threshold=0.0),
    )
    samples = (1 / 65536, 0.99999, -0.12345, 1.5)
    pcm = struct.pack("<4f", *samples)

    events = [event async for event in engine.stream(_chunks(pcm))]

    decoded_audio = model.transcribe_calls[0][0]
    assert list(decoded_audio) == pytest.approx(struct.unpack("<4f", pcm), rel=0, abs=0)
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=4 / 16_000),
    ]


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
async def test_ac_11_4_equivalent_s16le_and_f32le_match_decoder_and_boundaries(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Equivalent signed and float PCM produce equal decoder input and public events."""

    s16_engine, s16_model = await _prepared_engine(
        tmp_path / "s16",
        fake_runtime,
        input_format=EngineInputFormat.PCM_S16LE,
        boundary_settings=UtteranceBoundarySettings(
            silence_threshold_s=0.01, voice_threshold=500 / 32768
        ),
    )
    f32_engine, f32_model = await _prepared_engine(
        tmp_path / "f32",
        fake_runtime,
        input_format=EngineInputFormat.PCM_F32,
        boundary_settings=UtteranceBoundarySettings(
            silence_threshold_s=0.01, voice_threshold=500 / 32768
        ),
    )
    s16_model.responses.append([SimpleNamespace(text="same", start=0.0)])
    f32_model.responses.append([SimpleNamespace(text="same", start=0.0)])
    samples = (1024, -2048, 500, 0)
    s16_pcm = struct.pack("<4h", *samples)
    f32_pcm = struct.pack("<4f", *(sample / 32768 for sample in samples))

    s16_events = [event async for event in s16_engine.stream(_chunks(s16_pcm))]
    f32_events = [event async for event in f32_engine.stream(_chunks(f32_pcm))]

    assert list(s16_model.transcribe_calls[0][0]) == pytest.approx(
        list(f32_model.transcribe_calls[0][0]),
        rel=0,
        abs=0,
    )
    assert s16_events == f32_events


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_format", "incomplete_chunk"),
    [
        (EngineInputFormat.PCM_S16LE, b"\x00"),
        (EngineInputFormat.PCM_F32, b"\x00\x00\x00"),
    ],
)
async def test_ac_11_5_invalid_chunk_does_not_decode_or_advance_health(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
    incomplete_chunk: bytes,
) -> None:
    """An incomplete PCM sample fails before decoding or public health progress."""

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
        input_format=input_format,
    )

    await engine.prepare()
    fake_runtime.instances[0].transcribe_calls.clear()
    with pytest.raises(ConfigurationError, match=rf"{input_format.value}.*whole samples"):
        async for _ in engine.stream(_chunks(incomplete_chunk)):
            pass

    assert fake_runtime.instances[0].transcribe_calls == []
    assert engine.health().state is EngineState.READY
    assert engine.health().processed_pcm_chunks == 0
    assert engine.health().emitted_events == 0


@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_format",
    [EngineInputFormat.PCM_S16LE, EngineInputFormat.PCM_F32],
)
async def test_ac_11_10_prepare_supports_both_formats_without_public_counters(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    input_format: EngineInputFormat,
) -> None:
    """Preparation succeeds for either supported PCM format without public progress counts."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=1,
        input_format=input_format,
    )

    await engine.prepare()

    assert engine.health().state is EngineState.READY
    assert engine.health().processed_pcm_chunks == 0
    assert engine.health().emitted_events == 0


@pytest.mark.req("SP-LFFASTERWHISPER-005")
@pytest.mark.asyncio
async def test_ac_11_8_default_voice_gate_matches_legacy_amplitude_500(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """The default voice threshold opens speech at normalized amplitude 500 and not below it."""

    engine, model = await _prepared_engine(tmp_path, fake_runtime)

    below_events = [event async for event in engine.stream(_chunks(struct.pack("<h", 499)))]
    equal_events = [event async for event in engine.stream(_chunks(struct.pack("<h", 500)))]

    assert below_events == []
    assert equal_events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        UtteranceEnd(utterance_id=1, ts_offset_s=1 / 16_000),
    ]
    assert len(model.transcribe_calls) == 1


@pytest.mark.req("SP-LFFASTERWHISPER-002")
def test_ac_11_2_and_6_invalid_values_precede_blocked_optional_dependencies() -> None:
    """Invalid engine settings fail before unavailable optional runtime dependencies are imported."""

    code = r"""
import sys
from pathlib import Path

sys.modules["numpy"] = None
sys.modules["faster_whisper"] = None

import listenflow
from listenflow import ConfigurationError, EngineInputFormat, OfflineModelDirectorySource
from listenflow.faster_whisper import FasterWhisperEngine

source = OfflineModelDirectorySource(directory_path=str(Path("models/model").resolve()))

for kwargs, error_type in (
    ({"input_format": EngineInputFormat.WAV_PCM_S16LE}, ValueError),
    ({"boundary_settings": {"voice_threshold": 500}}, TypeError),
):
    try:
        FasterWhisperEngine(model_source=source, cpu_threads=1, **kwargs)
    except ConfigurationError as exc:
        assert isinstance(exc.__cause__, error_type)
    else:
        raise AssertionError(f"expected {error_type.__name__} for {kwargs!r}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.asyncio
async def test_ac_7_15_word_timestamps_changes_decoder_kwargs_not_public_shape(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Word timestamps affect decoder controls without changing public segment event shape."""

    mitigation = FasterWhisperSilenceMitigation(hallucination_silence_threshold_s=2.5)
    engine, model = await _prepared_engine(
        tmp_path,
        fake_runtime,
        word_timestamps=True,
        silence_mitigation=mitigation,
    )
    model.responses.append([SimpleNamespace(text="hello", start=0.25, words=[object()])])

    events = [event async for event in engine.stream(_chunks(_pcm_chunk(700, duration_s=0.01)))]

    kwargs = model.transcribe_calls[0][1]
    assert kwargs["word_timestamps"] is True
    assert kwargs["hallucination_silence_threshold"] == 2.5
    assert engine.capabilities.timestamp_detail == TimestampDetail.SEGMENT
    assert events == [
        UtteranceStart(utterance_id=1, ts_offset_s=0.0),
        Partial(utterance_id=1, ts_offset_s=0.25, text="hello"),
        Final(utterance_id=1, ts_offset_s=0.25, text="hello"),
        UtteranceEnd(utterance_id=1, ts_offset_s=0.01),
    ]
    assert all(not hasattr(event, "words") for event in events)


@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.skipif(
    not os.environ.get(_PINNED_MODEL_ENV),
    reason=f"set {_PINNED_MODEL_ENV} to run the pinned offline integration test",
)
@pytest.mark.asyncio
async def test_ac_7_2_ac_7_3_and_ac_7_14_pinned_asset_prepares_and_decodes_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verified offline model and audio assets support preparation and public CPU transcription."""

    model_directory = Path(os.environ[_PINNED_MODEL_ENV])
    _verify_pinned_model_directory(model_directory)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=str(model_directory)),
        cpu_threads=2,
        initial_prompt="ListenFlow",
        boundary_settings=UtteranceBoundarySettings(silence_threshold_s=0.1),
    )
    await engine.prepare()
    pcm = _read_verified_reference_audio()

    events = [event async for event in engine.stream(_chunks(pcm))]

    assert any(isinstance(event, Final) and event.text for event in events)
    bias = engine.health().bias_application
    assert bias is not None
    assert bias.applied is True
    assert bias.applied_value == "ListenFlow"


@pytest.mark.req("SP-LFFASTERWHISPER-010")
@pytest.mark.gpu
@pytest.mark.skipif(
    not (os.environ.get(_PINNED_MODEL_ENV) and os.environ.get(_CUDA_TEST_ENV) == "1"),
    reason=(f"set {_PINNED_MODEL_ENV} and {_CUDA_TEST_ENV}=1 to run the CUDA integration test"),
)
@pytest.mark.asyncio
async def test_ac_2_7_ac_2_8_ac_2_9_and_ac_6_7_cuda_extra_runs_real_gpu_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA extra supports verified offline model loading and complete public GPU transcription."""

    model_directory = Path(os.environ[_PINNED_MODEL_ENV])
    _verify_pinned_model_directory(model_directory)
    pcm = _read_verified_reference_audio()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=str(model_directory)),
        cpu_threads=2,
        device="cuda",
        compute_type="float16",
        beam_size=5,
    )
    await engine.prepare()
    status = engine.runtime_status()
    assert status is not None
    assert status.model_label == "small"
    assert status.model_directory_path == str(model_directory)
    assert status.requested_device == status.effective_device == "cuda"
    assert status.device_index == (0,)
    assert status.requested_compute_type == status.effective_compute_type == "float16"
    assert status.requested_cpu_threads == 2
    assert status.cuda is not None
    assert status.cuda.cublas_distribution_version == "12.9.2.10"
    assert status.cuda.cuda_nvrtc_distribution_version == "12.9.86"
    events = [event async for event in engine.stream(_chunks(pcm))]

    assert isinstance(events[0], UtteranceStart)
    assert isinstance(events[-1], UtteranceEnd)
    utterance_id = events[0].utterance_id
    assert events[-1].utterance_id == utterance_id
    transcript_events = events[1:-1]
    assert len(transcript_events) % 2 == 0
    final_texts: list[str] = []
    for partial, final in zip(transcript_events[::2], transcript_events[1::2], strict=True):
        assert isinstance(partial, Partial)
        assert isinstance(final, Final)
        assert partial.utterance_id == final.utterance_id == utterance_id
        assert partial.ts_offset_s == final.ts_offset_s
        assert final.text
        final_texts.append(final.text)
        assert partial.text == " ".join(final_texts)
    assert final_texts
    print(f"Measured CUDA transcript: {' '.join(final_texts)}")


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.asyncio
async def test_ac_8_1_prepare_rejects_a_mel_bin_mismatch_before_warm_up(
    tmp_path: Path, fake_runtime: type[FakeWhisperModel]
) -> None:
    """Preparation rejects a model and feature-extractor mismatch before warm-up."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(model_source=source, cpu_threads=3, model="large-v3")
    fake_runtime.runtime_profiles.append(
        SimpleNamespace(device="cpu", device_index=[0], compute_type="int8_float32", n_mels=128)
    )

    with pytest.raises(ConfigurationError, match=r"preprocessor_config\.json") as failure:
        await engine.prepare()

    assert "80" in str(failure.value) and "128" in str(failure.value)
    assert fake_runtime.instances[-1].transcribe_calls == []
    with pytest.raises(EngineStateError):
        engine.runtime_status()
    assert engine.health().state is EngineState.UNPREPARED

    await engine.prepare()

    status = engine.runtime_status()
    assert status is not None and status.model_label == "large-v3"
    assert engine.health().state is EngineState.READY


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.skipif(
    not os.environ.get(_LARGE_V3_MODEL_ENV),
    reason=f"set {_LARGE_V3_MODEL_ENV} to run the large-v3 CPU integration test",
)
@pytest.mark.asyncio
async def test_ac_8_2_large_v3_without_preprocessor_config_fails_at_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation rejects large-v3 weights lacking matching preprocessor configuration."""

    model_directory = Path(os.environ[_LARGE_V3_MODEL_ENV])
    _verify_pinned_model_directory(
        model_directory,
        revision=_LARGE_V3_REVISION,
        digests=_LARGE_V3_DIGESTS,
    )
    partial_directory = tmp_path / "large-v3-without-preprocessor-config"
    partial_directory.mkdir()
    for name in _LARGE_V3_DIGESTS:
        if name != "preprocessor_config.json":
            (partial_directory / name).symlink_to(model_directory / name)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=str(partial_directory)),
        cpu_threads=4,
        model="large-v3",
    )

    with pytest.raises(ConfigurationError, match=r"preprocessor_config\.json"):
        await engine.prepare()

    with pytest.raises(EngineStateError):
        engine.runtime_status()
    assert engine.health().state is EngineState.UNPREPARED


@pytest.mark.req("SP-LFFASTERWHISPER-004")
@pytest.mark.skipif(
    not os.environ.get(_LARGE_V3_MODEL_ENV),
    reason=f"set {_LARGE_V3_MODEL_ENV} to run the large-v3 CPU integration test",
)
@pytest.mark.asyncio
async def test_ac_5_3_large_v3_weights_prepare_and_decode_on_the_cpu_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verified large-v3 assets support offline CPU preparation and public transcription."""

    model_directory = Path(os.environ[_LARGE_V3_MODEL_ENV])
    _verify_pinned_model_directory(
        model_directory,
        revision=_LARGE_V3_REVISION,
        digests=_LARGE_V3_DIGESTS,
    )
    pcm = _read_verified_reference_audio()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=str(model_directory)),
        cpu_threads=4,
        model="large-v3",
    )
    await engine.prepare()
    events = [event async for event in engine.stream(_chunks(pcm))]

    final_texts = [event.text for event in events if isinstance(event, Final)]
    assert final_texts
    assert isinstance(events[0], UtteranceStart)
    assert isinstance(events[-1], UtteranceEnd)
    print(f"Measured large-v3 CPU transcript: {' '.join(final_texts)}")


@pytest.mark.req("SP-LFFASTERWHISPER-010")
@pytest.mark.gpu
@pytest.mark.skipif(
    not (os.environ.get(_LARGE_V3_MODEL_ENV) and os.environ.get(_CUDA_TEST_ENV) == "1"),
    reason=(
        f"set {_LARGE_V3_MODEL_ENV} and {_CUDA_TEST_ENV}=1 to run the large-v3 CUDA integration test"
    ),
)
@pytest.mark.asyncio
async def test_ac_5_4_and_ac_6_7_large_v3_cuda_float16_runs_real_gpu_inference_with_the_cuda_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA extra supports verified large-v3 loading and public GPU transcription without cuDNN."""

    model_directory = Path(os.environ[_LARGE_V3_MODEL_ENV])
    _verify_pinned_model_directory(
        model_directory,
        revision=_LARGE_V3_REVISION,
        digests=_LARGE_V3_DIGESTS,
    )
    pcm = _read_verified_reference_audio()

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=str(model_directory)),
        cpu_threads=4,
        model="large-v3",
        device="cuda",
        compute_type="float16",
        beam_size=1,
    )
    await engine.prepare()
    status = engine.runtime_status()
    assert status is not None
    assert status.model_label == "large-v3"
    assert status.model_directory_path == str(model_directory)
    assert status.requested_device == status.effective_device == "cuda"
    assert status.device_index == (0,)
    assert status.requested_compute_type == status.effective_compute_type == "float16"
    assert status.requested_cpu_threads == 4
    assert status.cuda is not None
    assert status.cuda.cublas_distribution_version == "12.9.2.10"
    assert status.cuda.cuda_nvrtc_distribution_version == "12.9.86"
    events = [event async for event in engine.stream(_chunks(pcm))]

    assert isinstance(events[0], UtteranceStart)
    assert isinstance(events[-1], UtteranceEnd)
    utterance_id = events[0].utterance_id
    assert events[-1].utterance_id == utterance_id
    transcript_events = events[1:-1]
    assert len(transcript_events) % 2 == 0
    final_texts: list[str] = []
    for partial, final in zip(transcript_events[::2], transcript_events[1::2], strict=True):
        assert isinstance(partial, Partial)
        assert isinstance(final, Final)
        assert partial.utterance_id == final.utterance_id == utterance_id
        assert final.text
        final_texts.append(final.text)
        assert partial.text == " ".join(final_texts)
    assert final_texts
    loaded_libraries = Path("/proc/self/maps").read_text()
    assert "libcudnn" not in loaded_libraries
    print(f"Measured large-v3 CUDA transcript: {' '.join(final_texts)}")


async def _assert_real_voiced_warmup(
    monkeypatch: pytest.MonkeyPatch, *, model: str, device: str, compute_type: str
) -> None:
    model_env = _PINNED_MODEL_ENV if model == "small" else _LARGE_V3_MODEL_ENV
    if not os.environ.get(model_env):
        pytest.skip(f"set {model_env} to run the offline warm-up integration test")
    model_directory = Path(os.environ[model_env])
    _verify_pinned_model_directory(
        model_directory,
        revision=_PINNED_REVISION if model == "small" else _LARGE_V3_REVISION,
        digests=_PINNED_DIGESTS if model == "small" else _LARGE_V3_DIGESTS,
    )
    pcm = _read_verified_reference_audio()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(
        model_source=OfflineModelDirectorySource(directory_path=str(model_directory)),
        model=model,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
    )
    real_model_cls = engine._model_cls
    warmup_segments: list[object] = []
    drained = False
    transcribe_calls = 0

    class ObservedModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.runtime = real_model_cls(*args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(self.runtime, name)

        def transcribe(self, *args: object, **kwargs: object) -> tuple[Iterator[object], object]:
            nonlocal transcribe_calls
            transcribe_calls += 1
            segments, info = self.runtime.transcribe(*args, **kwargs)
            if transcribe_calls != 1:
                return segments, info
            with pytest.raises(EngineStateError):
                engine.runtime_status()
            assert engine.health().state is EngineState.UNPREPARED

            def observe() -> Iterator[object]:
                nonlocal drained
                for segment in segments:
                    with pytest.raises(EngineStateError):
                        engine.runtime_status()
                    assert engine.health().state is EngineState.UNPREPARED
                    warmup_segments.append(segment)
                    yield segment
                with pytest.raises(EngineStateError):
                    engine.runtime_status()
                assert engine.health().state is EngineState.UNPREPARED
                drained = True

            return observe(), info

    engine._model_cls = ObservedModel
    await engine.prepare()
    assert transcribe_calls == 1
    assert drained, "prepare() must exhaust the warm-up iterator before returning"
    assert warmup_segments, "default VAD discarded the warm-up: no encoder/decoder evidence"
    assert engine.runtime_status() is not None
    assert engine.health().state is EngineState.READY
    events = [event async for event in engine.stream(_chunks(pcm))]
    assert any(isinstance(event, Final) and event.text for event in events)
    assert isinstance(events[0], UtteranceStart)
    assert isinstance(events[-1], UtteranceEnd)


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-009")
@pytest.mark.parametrize("model", ["small", "large-v3"])
async def test_ac_12_3_cpu_warmup_yields_segments_before_readiness(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """CPU preparation drains non-empty voiced warm-up segments before readiness for both models."""

    await _assert_real_voiced_warmup(monkeypatch, model=model, device="cpu", compute_type="int8")


@pytest.mark.req("SP-LFFASTERWHISPER-014")
@pytest.mark.req("SP-LFFASTERWHISPER-009")
@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get(_CUDA_TEST_ENV) != "1", reason=f"set {_CUDA_TEST_ENV}=1")
@pytest.mark.parametrize("model", ["small", "large-v3"])
async def test_ac_12_3_cuda_warmup_yields_segments_before_readiness(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """CUDA preparation drains non-empty voiced warm-up segments before readiness for both models."""

    await _assert_real_voiced_warmup(
        monkeypatch, model=model, device="cuda", compute_type="float16"
    )


_DECODE_LIMIT_KEYS = ("max_new_tokens", "repetition_penalty", "no_repeat_ngram_size")


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.asyncio
async def test_ac_7_1_decode_limits_default_to_upstream_and_are_not_forwarded(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Default decoding limits retain upstream behavior without forwarding explicit limits."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(model_source=source, cpu_threads=1)
    assert (engine.max_new_tokens, engine.repetition_penalty, engine.no_repeat_ngram_size) == (
        None,
        1.0,
        0,
    )
    await engine.prepare()
    model = fake_runtime.instances[-1]
    model.responses.append([SimpleNamespace(text="hello", start=0.0)])

    [_ async for _ in engine.stream(_chunks(_pcm_chunk(700, duration_s=0.01)))]

    assert len(model.transcribe_calls) == 2
    for _, kwargs in model.transcribe_calls:
        assert not set(_DECODE_LIMIT_KEYS) & set(kwargs)


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.asyncio
async def test_ac_7_2_decode_limits_reach_warm_up_and_every_transcription_unchanged(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
) -> None:
    """Selected decoding limits reach warm-up and every utterance unchanged."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(
        model_source=source,
        cpu_threads=1,
        max_new_tokens=64,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    await engine.prepare()
    model = fake_runtime.instances[-1]
    model.responses.append([SimpleNamespace(text="hello", start=0.0)])
    model.responses.append([SimpleNamespace(text="again", start=0.0)])

    [
        _
        async for _ in engine.stream(
            _chunks(
                _pcm_chunk(700, duration_s=0.01),
                _pcm_chunk(0, duration_s=1.0),
                _pcm_chunk(700, duration_s=0.01),
            )
        )
    ]

    assert len(model.transcribe_calls) == 3
    for _, kwargs in model.transcribe_calls:
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["repetition_penalty"] == 1.2
        assert kwargs["no_repeat_ngram_size"] == 3


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.parametrize(
    ("overrides", "forwarded"),
    [
        ({"max_new_tokens": None, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}, {}),
        ({"max_new_tokens": 448}, {"max_new_tokens": 448}),
        ({"repetition_penalty": 1.05}, {"repetition_penalty": 1.05}),
        ({"no_repeat_ngram_size": 2}, {"no_repeat_ngram_size": 2}),
    ],
)
@pytest.mark.asyncio
async def test_ac_7_3_only_non_default_decode_limits_are_forwarded(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    overrides: dict[str, object],
    forwarded: dict[str, object],
) -> None:
    """Only non-default decoding limits are forwarded to transcription."""

    source = _write_model_files(tmp_path / "model")
    engine = FasterWhisperEngine(model_source=source, cpu_threads=1, **overrides)  # type: ignore[arg-type]

    await engine.prepare()

    _, kwargs = fake_runtime.instances[-1].transcribe_calls[0]
    assert {key: kwargs[key] for key in _DECODE_LIMIT_KEYS if key in kwargs} == forwarded


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("max_new_tokens", True, TypeError, r"max_new_tokens.*None or a positive integer"),
        ("max_new_tokens", 1.5, TypeError, r"max_new_tokens.*None or a positive integer"),
        ("max_new_tokens", "64", TypeError, r"max_new_tokens.*None or a positive integer"),
        ("max_new_tokens", 0, ValueError, r"max_new_tokens.*None or a positive integer"),
        ("max_new_tokens", -1, ValueError, r"max_new_tokens.*None or a positive integer"),
        (
            "repetition_penalty",
            "1.2",
            TypeError,
            r"repetition_penalty.*finite number greater than 0",
        ),
        (
            "repetition_penalty",
            True,
            TypeError,
            r"repetition_penalty.*finite number greater than 0",
        ),
        (
            "repetition_penalty",
            None,
            TypeError,
            r"repetition_penalty.*finite number greater than 0",
        ),
        ("repetition_penalty", 0, ValueError, r"repetition_penalty.*finite number greater than 0"),
        (
            "repetition_penalty",
            -0.5,
            ValueError,
            r"repetition_penalty.*finite number greater than 0",
        ),
        (
            "repetition_penalty",
            float("nan"),
            ValueError,
            r"repetition_penalty.*finite number greater than 0",
        ),
        (
            "repetition_penalty",
            float("inf"),
            ValueError,
            r"repetition_penalty.*finite number greater than 0",
        ),
        ("no_repeat_ngram_size", True, TypeError, r"no_repeat_ngram_size.*integer 0 or greater"),
        ("no_repeat_ngram_size", 1.0, TypeError, r"no_repeat_ngram_size.*integer 0 or greater"),
        ("no_repeat_ngram_size", None, TypeError, r"no_repeat_ngram_size.*integer 0 or greater"),
        ("no_repeat_ngram_size", -1, ValueError, r"no_repeat_ngram_size.*integer 0 or greater"),
    ],
)
def test_ac_7_4_invalid_decode_limits_fail_before_runtime_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid decoding limits fail before the optional runtime loads."""

    monkeypatch.setattr(
        "listenflow.faster_whisper._load_faster_whisper_model",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loader must not run")),
    )

    with pytest.raises(ConfigurationError, match=message) as captured:
        FasterWhisperEngine(
            model_source=_write_model_files(tmp_path / "model"),
            cpu_threads=1,
            **{field: value},
        )
    assert isinstance(captured.value.__cause__, error_type)


@pytest.mark.req("SP-LFFASTERWHISPER-003")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_new_tokens", 1),
        ("max_new_tokens", 448),
        ("repetition_penalty", 1),
        ("repetition_penalty", 2.5),
        ("no_repeat_ngram_size", 0),
        ("no_repeat_ngram_size", 4),
    ],
)
def test_ac_7_4_valid_decode_limits_are_retained(
    tmp_path: Path,
    fake_runtime: type[FakeWhisperModel],
    field: str,
    value: object,
) -> None:
    """Valid decoding limits remain available as configured."""

    engine = FasterWhisperEngine(
        model_source=_write_model_files(tmp_path / "model"),
        cpu_threads=1,
        **{field: value},
    )

    assert getattr(engine, field) == value
