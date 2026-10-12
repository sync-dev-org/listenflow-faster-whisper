"""Language values supported by an offline faster-whisper model directory."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Only languages recognized by faster-whisper 1.2.1 are listed here. The names
# follow Whisper's language names; the three-letter forms are ISO 639-3 codes.
# Javanese uses the contract code jv, while faster-whisper uses jw.
_LANGUAGES: dict[str, tuple[str, str]] = {
    "af": ("afrikaans", "afr"),
    "am": ("amharic", "amh"),
    "ar": ("arabic", "ara"),
    "as": ("assamese", "asm"),
    "az": ("azerbaijani", "aze"),
    "ba": ("bashkir", "bak"),
    "be": ("belarusian", "bel"),
    "bg": ("bulgarian", "bul"),
    "bn": ("bengali", "ben"),
    "bo": ("tibetan", "bod"),
    "br": ("breton", "bre"),
    "bs": ("bosnian", "bos"),
    "ca": ("catalan", "cat"),
    "cs": ("czech", "ces"),
    "cy": ("welsh", "cym"),
    "da": ("danish", "dan"),
    "de": ("german", "deu"),
    "el": ("greek", "ell"),
    "en": ("english", "eng"),
    "es": ("spanish", "spa"),
    "et": ("estonian", "est"),
    "eu": ("basque", "eus"),
    "fa": ("persian", "fas"),
    "fi": ("finnish", "fin"),
    "fo": ("faroese", "fao"),
    "fr": ("french", "fra"),
    "gl": ("galician", "glg"),
    "gu": ("gujarati", "guj"),
    "ha": ("hausa", "hau"),
    "haw": ("hawaiian", "haw"),
    "he": ("hebrew", "heb"),
    "hi": ("hindi", "hin"),
    "hr": ("croatian", "hrv"),
    "ht": ("haitian creole", "hat"),
    "hu": ("hungarian", "hun"),
    "hy": ("armenian", "hye"),
    "id": ("indonesian", "ind"),
    "is": ("icelandic", "isl"),
    "it": ("italian", "ita"),
    "ja": ("japanese", "jpn"),
    "jv": ("javanese", "jav"),
    "ka": ("georgian", "kat"),
    "kk": ("kazakh", "kaz"),
    "km": ("khmer", "khm"),
    "kn": ("kannada", "kan"),
    "ko": ("korean", "kor"),
    "la": ("latin", "lat"),
    "lb": ("luxembourgish", "ltz"),
    "ln": ("lingala", "lin"),
    "lo": ("lao", "lao"),
    "lt": ("lithuanian", "lit"),
    "lv": ("latvian", "lav"),
    "mg": ("malagasy", "mlg"),
    "mi": ("maori", "mri"),
    "mk": ("macedonian", "mkd"),
    "ml": ("malayalam", "mal"),
    "mn": ("mongolian", "mon"),
    "mr": ("marathi", "mar"),
    "ms": ("malay", "msa"),
    "mt": ("maltese", "mlt"),
    "my": ("myanmar", "mya"),
    "ne": ("nepali", "nep"),
    "nl": ("dutch", "nld"),
    "nn": ("nynorsk", "nno"),
    "no": ("norwegian", "nor"),
    "oc": ("occitan", "oci"),
    "pa": ("punjabi", "pan"),
    "pl": ("polish", "pol"),
    "ps": ("pashto", "pus"),
    "pt": ("portuguese", "por"),
    "ro": ("romanian", "ron"),
    "ru": ("russian", "rus"),
    "sa": ("sanskrit", "san"),
    "sd": ("sindhi", "snd"),
    "si": ("sinhala", "sin"),
    "sk": ("slovak", "slk"),
    "sl": ("slovenian", "slv"),
    "sn": ("shona", "sna"),
    "so": ("somali", "som"),
    "sq": ("albanian", "sqi"),
    "sr": ("serbian", "srp"),
    "su": ("sundanese", "sun"),
    "sv": ("swedish", "swe"),
    "sw": ("swahili", "swa"),
    "ta": ("tamil", "tam"),
    "te": ("telugu", "tel"),
    "tg": ("tajik", "tgk"),
    "th": ("thai", "tha"),
    "tk": ("turkmen", "tuk"),
    "tl": ("tagalog", "tgl"),
    "tr": ("turkish", "tur"),
    "tt": ("tatar", "tat"),
    "uk": ("ukrainian", "ukr"),
    "ur": ("urdu", "urd"),
    "uz": ("uzbek", "uzb"),
    "vi": ("vietnamese", "vie"),
    "yi": ("yiddish", "yid"),
    "yo": ("yoruba", "yor"),
    "yue": ("cantonese", "yue"),
    "zh": ("chinese", "zho"),
}
_UPSTREAM_CODES = frozenset("jw" if code == "jv" else code for code in _LANGUAGES)
_TOKEN_PATTERN = re.compile(r"<\|([a-z]{2,3})\|>")


def _read_json(directory: Path, name: str) -> object:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"model_source.directory_path is missing required file: {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model_source.directory_path has unreadable {name}") from exc


def _text_vocabulary_has_empty_token(content: bytes) -> bool:
    # CTranslate2 splits on LF and strips CR only when every token ends in CR.
    if not content:
        return False
    tokens = content.split(b"\n")
    if content.endswith(b"\n"):
        tokens.pop()
    if tokens and all(token.endswith(b"\r") for token in tokens):
        tokens = [token[:-1] for token in tokens]
    return b"" in tokens


def model_language_metadata(directory_path: str) -> tuple[frozenset[str], bool]:
    """Read the vocabulary rule and the conversion's usable language IDs."""

    directory = Path(directory_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"model_source.directory_path is not a directory: {directory}")
    config = _read_json(directory, "config.json")
    tokenizer = _read_json(directory, "tokenizer.json")
    vocabulary = next(
        (
            directory / name
            for name in ("vocabulary.json", "vocabulary.txt")
            if (directory / name).exists()
        ),
        None,
    )
    if vocabulary is None:
        raise FileNotFoundError(
            "model_source.directory_path is missing required file: vocabulary.json or vocabulary.txt"
        )
    if not vocabulary.is_file():
        raise FileNotFoundError(
            f"model_source.directory_path is missing required file: {vocabulary.name}"
        )
    try:
        if vocabulary.suffix == ".json":
            tokens = json.loads(vocabulary.read_text(encoding="utf-8"))
            if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                raise ValueError(f"model_source.directory_path has invalid {vocabulary.name}")
            multilingual = "" in tokens
        else:
            multilingual = _text_vocabulary_has_empty_token(vocabulary.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model_source.directory_path has unreadable {vocabulary.name}") from exc
    if not multilingual:
        return frozenset({"en"}), False

    lang_ids = config.get("lang_ids") if isinstance(config, dict) else None
    if (
        not isinstance(lang_ids, list)
        or not lang_ids
        or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in lang_ids)
    ):
        raise ValueError("model_source.directory_path has invalid config.json lang_ids")
    added_tokens = tokenizer.get("added_tokens") if isinstance(tokenizer, dict) else None
    if not isinstance(added_tokens, list):
        raise ValueError(  # noqa: TRY004 - malformed metadata is a configuration value
            "model_source.directory_path has invalid tokenizer.json added_tokens"
        )
    id_to_token = {
        token.get("id"): token.get("content") for token in added_tokens if isinstance(token, dict)
    }
    supported: set[str] = set()
    for token_id in lang_ids:
        token = id_to_token.get(token_id)
        match = _TOKEN_PATTERN.fullmatch(token) if isinstance(token, str) else None
        if match is None:
            raise ValueError("model_source.directory_path has invalid config.json lang_ids")
        upstream_code = match.group(1)
        if upstream_code in _UPSTREAM_CODES:
            supported.add("jv" if upstream_code == "jw" else upstream_code)
    if not supported:
        raise ValueError("model_source.directory_path has no usable config.json lang_ids")
    return frozenset(supported), True


def validate_language(value: str | None, supported: frozenset[str]) -> None:
    """Reject unsupported notation and suggest only a model-supported code."""

    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError("language must be a contract code or None")
    if value in supported:
        return
    base = re.split("[-_]", value, maxsplit=1)[0].lower()
    hints = {alias: code for code in supported for alias in (code, *_LANGUAGES[code])}
    accepted = ", ".join(sorted(supported))
    hint = hints.get(base)
    suffix = f"; use '{hint}'" if hint is not None else ""
    raise ValueError(
        f"language {value!r} is not accepted; accepted language codes: {accepted}{suffix}"
    )
