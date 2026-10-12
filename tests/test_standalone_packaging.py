from __future__ import annotations

import ast
import copy
import email
import email.message
import importlib
import importlib.metadata
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tomllib
import wave
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYPI_REGISTRY_URL = "https://pypi.org/simple"
CORE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
PROJECT_DEPENDENCIES = {
    "faster-whisper==1.2.1",
    "listenflow>=0.17.0",
}
CUDA_EXTRA_NAME = "cuda"
CUDA_EXTRA_DEPENDENCIES = {
    "nvidia-cublas-cu12>=12.9.1.4,<12.10",
}
CUDA_LOCKED_VERSIONS = {
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
    "ctranslate2": "4.8.2",
}
CUDA_LOCKED_PACKAGE_DEPENDENCIES = {
    "nvidia-cublas-cu12": {"nvidia-cuda-nvrtc-cu12"},
}
MODULE_EXPORTS = {
    "FasterWhisperCudaRuntimeStatus",
    "FasterWhisperEngine",
    "FasterWhisperRuntimeStatus",
    "FasterWhisperSilenceMitigation",
    "__version__",
}


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _locked_package(lock: dict[str, Any], name: str) -> dict[str, Any]:
    packages = [package for package in lock["package"] if package["name"] == name]
    assert len(packages) == 1
    return packages[0]


def _git_identity(source: str) -> tuple[str, str, str]:
    source_without_commit, separator, commit = source.rpartition("#")
    assert separator
    parsed = urlsplit(source_without_commit)
    revisions = parse_qs(parsed.query).get("rev")
    assert revisions is not None and len(revisions) == 1
    repository = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return repository, revisions[0], commit


def _assert_registry_locked_core(
    lock: dict[str, Any], *, registry_url: str = PYPI_REGISTRY_URL
) -> dict[str, Any]:
    core = _locked_package(lock, "listenflow")
    assert core["source"] == {"registry": registry_url}
    return core


def _assert_standalone_metadata(project: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    assert set(project["project"]["dependencies"]) == PROJECT_DEPENDENCIES
    uv_settings = project.get("tool", {}).get("uv", {})
    assert "listenflow" not in uv_settings.get("sources", {})
    assert "prerelease" not in uv_settings
    development_core = [
        requirement
        for requirement in project["dependency-groups"]["dev"]
        if Requirement(requirement).name == "listenflow"
    ]
    assert not development_core
    return _assert_registry_locked_core(lock)


def _isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    return environment


def _run(*command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=_isolated_environment(),
        check=True,
        capture_output=True,
        text=True,
    )


def _git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return _run("git", *arguments, cwd=cwd)


def _commit_fixture_version(repository: Path, name: str, version: str) -> str:
    package = repository / "src" / name
    package.mkdir(parents=True, exist_ok=True)
    (repository / "pyproject.toml").write_text(
        f"""\
[project]
name = "{name}"
version = "{version}"
requires-python = ">=3.12,<3.14"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
""",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    _git("add", "pyproject.toml", f"src/{name}/__init__.py", cwd=repository)
    _git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--quiet",
        "-m",
        f"fixture {name} {version}",
        cwd=repository,
    )
    return _git("rev-parse", "HEAD", cwd=repository).stdout.strip()


def _publish_fixture_version(repository: Path, index: Path, name: str, version: str) -> None:
    _commit_fixture_version(repository, name, version)
    package_index = index / name
    _run("uv", "build", "--wheel", "--out-dir", str(package_index), cwd=repository)
    (package_index / "index.html").write_text(
        "\n".join(
            f'<a href="{wheel.name}">{wheel.name}</a>'
            for wheel in sorted(package_index.glob("*.whl"))
        ),
        encoding="utf-8",
    )


def _write_fixture_module(repository: Path) -> None:
    package = repository / "src/fixture_module"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        """\
[project]
name = "fixture-module"
version = "1.0.0"
requires-python = ">=3.12,<3.14"
dependencies = [
    "idna>=3.19",
    "listenflow>=0.17.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fixture_module"]

""",
        encoding="utf-8",
    )


def _commit_module_copy(repository: Path) -> str:
    """Commit this repository's tracked project files into a throwaway Git remote."""
    repository.mkdir()
    _git("init", "--quiet", "--initial-branch=main", str(repository), cwd=repository.parent)
    shutil.copy(REPOSITORY_ROOT / "pyproject.toml", repository / "pyproject.toml")
    shutil.copytree(
        REPOSITORY_ROOT / "src",
        repository / "src",
        ignore=shutil.ignore_patterns("__pycache__", "_version.py"),
    )
    _git("add", "pyproject.toml", "src", cwd=repository)
    _git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--quiet",
        "-m",
        "module copy",
        cwd=repository,
    )
    return _git("rev-parse", "HEAD", cwd=repository).stdout.strip()


def _write_consumer(
    repository: Path, *, core_remote: str, core_revision: str, module_remote: str
) -> None:
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        f"""\
[project]
name = "consumer"
version = "1.0.0"
requires-python = ">=3.12,<3.14"
dependencies = [
    "listenflow",
    "listenflow-faster-whisper",
]

[tool.uv.sources]
listenflow = {{ git = "{core_remote}", rev = "{core_revision}" }}
listenflow-faster-whisper = {{ git = "{module_remote}" }}
""",
        encoding="utf-8",
    )


def _wheel_metadata(wheel: Path) -> tuple[email.message.Message, set[str]]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_path))
        files = set(archive.namelist())
    return metadata, files


def _package_and_specifier(requirement: Requirement) -> tuple[str, SpecifierSet]:
    return canonicalize_name(requirement.name), requirement.specifier


def _declared_requirements(requirements: set[str]) -> set[tuple[str, SpecifierSet]]:
    return {_package_and_specifier(Requirement(requirement)) for requirement in requirements}


def _assert_cuda_extra_contract(project: dict[str, Any], lock: dict[str, Any]) -> None:
    declared = [
        Requirement(requirement)
        for requirement in project["project"]["optional-dependencies"][CUDA_EXTRA_NAME]
    ]
    assert len(declared) == len(CUDA_EXTRA_DEPENDENCIES)
    for requirement in declared:
        assert requirement.url is None
        assert not requirement.extras
        assert requirement.marker is None
    assert {_package_and_specifier(requirement) for requirement in declared} == (
        _declared_requirements(CUDA_EXTRA_DEPENDENCIES)
    )

    module = _locked_package(lock, "listenflow-faster-whisper")
    assert set(module["optional-dependencies"]) == {CUDA_EXTRA_NAME}
    assert {
        canonicalize_name(dependency["name"])
        for dependency in module["optional-dependencies"][CUDA_EXTRA_NAME]
    } == {name for name, _ in _declared_requirements(CUDA_EXTRA_DEPENDENCIES)}
    assert module["metadata"]["provides-extras"] == [CUDA_EXTRA_NAME]
    marked_rows = [
        row for row in module["metadata"]["requires-dist"] if row.get("marker") is not None
    ]
    assert all(row["marker"] == f"extra == '{CUDA_EXTRA_NAME}'" for row in marked_rows)
    assert {
        (canonicalize_name(row["name"]), SpecifierSet(row["specifier"])) for row in marked_rows
    } == _declared_requirements(CUDA_EXTRA_DEPENDENCIES)

    for name, version in CUDA_LOCKED_VERSIONS.items():
        assert _locked_package(lock, name)["version"] == version
    for requirement in declared:
        locked_version = _locked_package(lock, canonicalize_name(requirement.name))["version"]
        assert requirement.specifier.contains(Version(locked_version), prereleases=False)
    for name, dependencies in CUDA_LOCKED_PACKAGE_DEPENDENCIES.items():
        assert {
            canonicalize_name(dependency["name"])
            for dependency in _locked_package(lock, name)["dependencies"]
        } == dependencies


@pytest.mark.req("SP-LFFASTERWHISPER-012")
@pytest.mark.req("SP-LFFASTERWHISPER-013")
@pytest.mark.req("SP-LFFASTERWHISPER-019")
def test_standalone_metadata_and_lock_pin_a_pypi_core() -> None:
    """Runtime metadata keeps a compatible core range and the development lock pins a PyPI release."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")

    core = _assert_standalone_metadata(project, lock)

    compatibility = next(
        Requirement(dependency)
        for dependency in project["project"]["dependencies"]
        if Requirement(dependency).name == "listenflow"
    )
    assert Version(core["version"]) in compatibility.specifier


@pytest.mark.req("SP-LFFASTERWHISPER-012")
@pytest.mark.req("SP-LFFASTERWHISPER-019")
def test_metadata_contract_rejects_a_dev_group_with_a_core_git_reference(tmp_path: Path) -> None:
    """Development dependencies cannot override the published core with a Git reference."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")
    git_reference = copy.deepcopy(project)
    git_reference["dependency-groups"]["dev"].append(
        f"listenflow @ git+{(tmp_path / 'listenflow.git').as_uri()}@main"
    )

    with pytest.raises(AssertionError):
        _assert_standalone_metadata(git_reference, lock)


@pytest.mark.req("SP-LFFASTERWHISPER-012")
def test_metadata_contract_rejects_a_project_level_core_git_source(tmp_path: Path) -> None:
    """Project-level core Git sources are rejected so consumers control their core source."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")
    leaking_source = copy.deepcopy(project)
    leaking_source.setdefault("tool", {}).setdefault("uv", {})["sources"] = {
        "listenflow": {"git": (tmp_path / "listenflow.git").as_uri(), "branch": "main"},
    }

    with pytest.raises(AssertionError):
        _assert_standalone_metadata(leaking_source, lock)


@pytest.mark.req("SP-LFFASTERWHISPER-012")
def test_installed_core_matches_the_locked_pypi_version() -> None:
    """The installed core version matches the release pinned from PyPI."""

    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")
    core = _assert_registry_locked_core(lock)
    distribution = importlib.metadata.distribution("listenflow")

    assert distribution.version == core["version"]
    assert distribution.read_text("direct_url.json") is None
    listenflow = importlib.import_module("listenflow")
    installed_package = Path(distribution.locate_file("listenflow")).resolve()
    assert Path(listenflow.__file__).resolve().is_relative_to(installed_package)


@pytest.mark.req("SP-LFFASTERWHISPER-011")
def test_cuda_extra_metadata_and_lock_match_the_measured_contract() -> None:
    """The CUDA extra declaration and lock match the measured runtime package set."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")

    _assert_cuda_extra_contract(project, lock)


@pytest.mark.req("SP-LFFASTERWHISPER-011")
@pytest.mark.parametrize(
    "cuda_extra",
    [
        pytest.param(["nvidia-cublas-cu12==12.9.2.10"], id="exact-pin-of-the-measured-version"),
        pytest.param(["nvidia-cublas-cu12==12.9.1.4"], id="exact-pin-of-another-version"),
        pytest.param(["nvidia-cublas-cu12>=12.9"], id="unbounded-range"),
        pytest.param(
            ["nvidia-cublas-cu12>=12.9.0.13,<12.9.2"], id="range-excluding-the-measured-version"
        ),
        pytest.param(["nvidia-cudnn-cu12==9.26.0.51"], id="other-package"),
        pytest.param(
            ["nvidia-cublas-cu12>=12.9.1.4,<12.10", "nvidia-cudnn-cu12==9.26.0.51"],
            id="additional-package",
        ),
        pytest.param([], id="empty"),
    ],
)
def test_cuda_extra_contract_rejects_a_project_declaration_outside_the_measured_set(
    cuda_extra: list[str],
) -> None:
    """The CUDA extra rejects declarations outside its measured package range and set."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")
    mutated = copy.deepcopy(project)
    mutated["project"]["optional-dependencies"][CUDA_EXTRA_NAME] = cuda_extra

    with pytest.raises(AssertionError):
        _assert_cuda_extra_contract(mutated, lock)


@pytest.mark.req("SP-LFFASTERWHISPER-011")
@pytest.mark.parametrize(
    "package_name",
    ["nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu12", "ctranslate2"],
)
def test_cuda_extra_contract_rejects_a_lock_with_another_cuda_runtime_version(
    package_name: str,
) -> None:
    """The CUDA extra rejects a lock with unmeasured runtime package versions."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")
    mutated = copy.deepcopy(lock)
    _locked_package(mutated, package_name)["version"] = "0.0.0"

    with pytest.raises(AssertionError):
        _assert_cuda_extra_contract(project, mutated)


@pytest.mark.req("SP-LFFASTERWHISPER-011")
def test_cuda_extra_contract_rejects_a_lock_whose_extra_projection_drifts() -> None:
    """The CUDA extra rejects lock metadata that disagrees with its declared dependencies."""

    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    lock = _read_toml(REPOSITORY_ROOT / "uv.lock")
    module = _locked_package(lock, "listenflow-faster-whisper")
    drifted_specifier = copy.deepcopy(lock)
    for row in _locked_package(drifted_specifier, "listenflow-faster-whisper")["metadata"][
        "requires-dist"
    ]:
        if row.get("marker") is not None:
            row["specifier"] = "==12.9.2.10"
    dropped_dependency = copy.deepcopy(lock)
    _locked_package(dropped_dependency, "nvidia-cublas-cu12")["dependencies"] = []
    assert module["metadata"]["provides-extras"] == [CUDA_EXTRA_NAME]

    with pytest.raises(AssertionError):
        _assert_cuda_extra_contract(project, drifted_specifier)
    with pytest.raises(AssertionError):
        _assert_cuda_extra_contract(project, dropped_dependency)


@pytest.mark.req("SP-LFFASTERWHISPER-017")
@pytest.mark.parametrize(
    ("tag", "expected_version"),
    [
        pytest.param("v9.8.7", "9.8.7", id="release-tag"),
        pytest.param("listenflow-faster-whisper/v9.8.7", "0.0.0.dev0", id="historical-tag"),
        pytest.param(None, "0.0.0.dev0", id="untagged-checkout"),
    ],
)
def test_wheel_version_uses_exact_release_tags_or_falls_back(
    tmp_path: Path, tag: str | None, expected_version: str
) -> None:
    """Exact vX.Y.Z tags set wheel release versions; other checkouts use 0.0.0.dev0."""

    checkout = tmp_path / "module"
    _commit_module_copy(checkout)
    if tag is not None:
        _git("tag", tag, cwd=checkout)
    wheelhouse = tmp_path / "wheelhouse"

    _run("uv", "build", "--offline", "--wheel", "--out-dir", str(wheelhouse), cwd=checkout)

    wheel = next(wheelhouse.glob("listenflow_faster_whisper-*.whl"))
    metadata, _ = _wheel_metadata(wheel)
    assert metadata["Version"] == expected_version


@pytest.mark.req("SP-LFFASTERWHISPER-001")
def test_module_wheel_owns_only_its_subtree_and_public_dependencies(tmp_path: Path) -> None:
    """The wheel owns only the engine import subtree and public runtime dependencies."""

    wheelhouse = tmp_path / "wheelhouse"
    _run(
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(wheelhouse),
        cwd=REPOSITORY_ROOT,
    )
    wheel = next(wheelhouse.glob("listenflow_faster_whisper-*.whl"))
    metadata, files = _wheel_metadata(wheel)
    requirements = metadata.get_all("Requires-Dist", [])

    parsed_requirements = [Requirement(requirement) for requirement in requirements]
    unmarked_requirements = {
        _package_and_specifier(requirement)
        for requirement in parsed_requirements
        if requirement.marker is None
    }
    marked_requirements = [
        requirement for requirement in parsed_requirements if requirement.marker is not None
    ]
    assert unmarked_requirements == {
        _package_and_specifier(Requirement(requirement)) for requirement in PROJECT_DEPENDENCIES
    }
    assert all(
        str(requirement.marker) == f'extra == "{CUDA_EXTRA_NAME}"'
        for requirement in marked_requirements
    )
    assert {
        _package_and_specifier(requirement) for requirement in marked_requirements
    } == _declared_requirements(CUDA_EXTRA_DEPENDENCIES)
    assert metadata.get_all("Provides-Extra", []) == [CUDA_EXTRA_NAME]
    assert all(
        marker not in requirement.lower()
        for requirement in requirements
        for marker in ("git+", "github.com", "branch=", "file:", "../")
    )
    assert not any(CORE_COMMIT_PATTERN.search(requirement) for requirement in requirements)
    package_files = {path for path in files if path.startswith("listenflow/")}
    assert package_files
    assert "listenflow/__init__.py" not in package_files
    assert "listenflow/faster_whisper/__init__.py" in package_files
    assert all(path.startswith("listenflow/faster_whisper/") for path in package_files)


@pytest.mark.req("SP-LFFASTERWHISPER-018")
def test_sdist_and_public_snapshot_exclude_the_same_development_files() -> None:
    """The source distribution and public snapshot exclude the same development files."""

    configuration = REPOSITORY_ROOT / "noren.yaml"
    if not configuration.is_file() and not (REPOSITORY_ROOT / "AGENTS.md").is_file():
        pytest.skip("the public snapshot does not include development publication settings")
    assert configuration.is_file(), "the development repository must include noren.yaml"
    publication = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    project = _read_toml(REPOSITORY_ROOT / "pyproject.toml")
    sdist = project["tool"]["hatch"]["build"]["targets"].get("sdist", {})

    assert set(sdist.get("exclude", [])) == set(publication["exclude"].splitlines())


@pytest.mark.req("SP-LFFASTERWHISPER-019")
def test_sdist_contains_distributed_files_without_development_files(tmp_path: Path) -> None:
    """The source distribution includes package files and keeps development files private."""

    distributions = tmp_path / "dist"
    _run("uv", "build", "--sdist", "--out-dir", str(distributions), cwd=REPOSITORY_ROOT)
    sdist = next(distributions.glob("listenflow_faster_whisper-*.tar.gz"))
    with tarfile.open(sdist) as archive:
        roots = {Path(name).parts[1] for name in archive.getnames() if len(Path(name).parts) > 1}

    assert {"src", "tests", "README.md", "LICENSE", "pyproject.toml"} <= roots
    assert {
        ".nf",
        ".github",
        "AGENTS.md",
        "CLAUDE.md",
        "tools",
        "noren.yaml",
        "noren-words.txt",
    }.isdisjoint(roots)


@pytest.mark.req("SP-LFFASTERWHISPER-020")
def test_distributions_declare_mit_and_include_the_license(tmp_path: Path) -> None:
    """The wheel declares MIT and both distributions include the complete MIT license."""

    expected_license = """\
MIT License

Copyright (c) 2026 minamik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    distributions = tmp_path / "dist"
    _run("uv", "build", "--out-dir", str(distributions), cwd=REPOSITORY_ROOT)
    wheel = next(distributions.glob("listenflow_faster_whisper-*.whl"))
    metadata, files = _wheel_metadata(wheel)

    assert metadata["License-Expression"] == "MIT"
    assert metadata.get_all("License-File") == ["LICENSE"]
    license_paths = [path for path in files if path.endswith(".dist-info/licenses/LICENSE")]
    assert len(license_paths) == 1
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(license_paths[0]).decode("utf-8") == expected_license

    sdist = next(distributions.glob("listenflow_faster_whisper-*.tar.gz"))
    with tarfile.open(sdist) as archive:
        license_paths = [
            name for name in archive.getnames() if Path(name).parts[1:] == ("LICENSE",)
        ]
        assert len(license_paths) == 1
        license_file = archive.extractfile(license_paths[0])
        assert license_file is not None
        assert license_file.read().decode("utf-8") == expected_license


@pytest.fixture(scope="module")
def warmup_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    wheelhouse = tmp_path_factory.mktemp("warmup-wheel")
    _run("uv", "build", "--wheel", "--out-dir", str(wheelhouse), cwd=REPOSITORY_ROOT)
    return next(wheelhouse.glob("listenflow_faster_whisper-*.whl"))


@pytest.mark.req("SP-LFFASTERWHISPER-009")
def test_ac_12_2_wheel_contains_speech_and_attribution(warmup_wheel: Path) -> None:
    """The wheel includes one voiced warm-up asset and its source and license attribution."""

    with zipfile.ZipFile(warmup_wheel) as archive:
        package_files = {
            member.filename: archive.read(member)
            for member in archive.infolist()
            if member.filename.startswith("listenflow/faster_whisper/")
            and not member.is_dir()
            and Path(member.filename).suffix not in {".py", ".pyc"}
            and "__pycache__" not in Path(member.filename).parts
            and not any(part.endswith(".dist-info") for part in Path(member.filename).parts)
        }
        speech_files = []
        for name, content in package_files.items():
            try:
                with wave.open(io.BytesIO(content)) as audio:
                    if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
                        16_000,
                        1,
                        2,
                    ):
                        continue
                    pcm = audio.readframes(audio.getnframes())
            except (wave.Error, EOFError):
                if content[8:12] == b"WAVE":
                    continue
                pcm = content
            if len(pcm) == 32_000 and any(sample for (sample,) in struct.iter_unpack("<h", pcm)):
                speech_files.append(name)
        assert len(speech_files) == 1, "wheel must ship one short speech asset in its owned subtree"
        speech = speech_files[0]
        notes = []
        for name, content in package_files.items():
            if name != speech and Path(name).parent == Path(speech).parent:
                try:
                    notes.append(content.decode("utf-8"))
                except UnicodeDecodeError:
                    continue
        required = (
            "LibriSpeech",
            "SLR12",
            "libri1",
            "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        )
        matching = [note for note in notes if all(token in note for token in required)]
        assert matching, "speech data must have adjacent source and license attribution"
        assert any(
            "crop" in note.lower()
            and re.search(r"16\s*(?:khz|,?000\s*hz)", note.lower())
            and "mono" in note.lower()
            and ("s16le" in note.lower() or "16-bit little-endian" in note.lower())
            and ("resampl" in note.lower() or "conver" in note.lower())
            for note in matching
        ), "attribution must describe the crop and 16 kHz mono s16le conversion"


@pytest.mark.req("SP-LFFASTERWHISPER-001")
@pytest.mark.req("SP-LFFASTERWHISPER-008")
@pytest.mark.skipif(
    not os.environ.get("LISTENFLOW_FASTER_WHISPER_MODEL_DIR"),
    reason="set LISTENFLOW_FASTER_WHISPER_MODEL_DIR for installed-wheel offline preparation",
)
def test_ac_12_2_installed_wheel_prepares_offline_outside_checkout(
    tmp_path: Path, warmup_wheel: Path
) -> None:
    """An installed wheel prepares offline from packaged speech outside the source checkout."""

    from test_faster_whisper_engine import _verify_pinned_model_directory

    model_directory = Path(os.environ["LISTENFLOW_FASTER_WHISPER_MODEL_DIR"])
    _verify_pinned_model_directory(model_directory)
    environment = tmp_path / "installed"
    _run("uv", "venv", "--python", sys.executable, str(environment), cwd=tmp_path)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = tmp_path / "requirements.txt"
    _run(
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--output-file",
        str(requirements),
        cwd=REPOSITORY_ROOT,
    )
    _run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--requirement",
        str(requirements),
        str(warmup_wheel),
        cwd=tmp_path,
    )
    script = r"""
import asyncio
import io
import os
from pathlib import Path
import re
import sys
import wave

import numpy as np

checkout = Path(sys.argv[1]).resolve()
assert not Path.cwd().resolve().is_relative_to(checkout)
model_directory = Path(sys.argv[2])
opened = set()
network_attempts = []
preparing = False

def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        assert not path.is_relative_to(checkout), f"checkout dependency: {path}"
        opened.add(path)
    if preparing and event in {"socket.__new__", "socket.connect", "socket.getaddrinfo"}:
        network_attempts.append(event)
        raise AssertionError(f"network access during prepare: {event}")

sys.addaudithook(audit)
os.environ["HF_HUB_OFFLINE"] = "1"
from listenflow import OfflineModelDirectorySource
import listenflow.faster_whisper as module

package = Path(module.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve())
engine = module.FasterWhisperEngine(
    model_source=OfflineModelDirectorySource(directory_path=str(model_directory)),
    cpu_threads=2,
)
real_model_cls = engine._model_cls
inputs = []

class RecordingModel:
    def __init__(self, *args, **kwargs):
        self.runtime = real_model_cls(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def transcribe(self, audio, **kwargs):
        inputs.append(audio.copy())
        return self.runtime.transcribe(audio, **kwargs)

engine._model_cls = RecordingModel
assets_before = set(model_directory.rglob("*"))

async def prepare():
    global preparing
    preparing = True
    try:
        await engine.prepare()
    finally:
        preparing = False

asyncio.run(prepare())
assert not network_attempts, network_attempts
assert len(inputs) == 1
assert any(inputs[0]), "installed preparation input is silent"
assert len(inputs[0]) == 16_000
speech_files = []
for path in tuple(opened):
    if (
        not path.is_relative_to(package)
        or path.suffix in {".py", ".pyc"}
        or "__pycache__" in path.relative_to(package).parts
        or any(part.endswith(".dist-info") for part in path.relative_to(package).parts)
        or not path.is_file()
    ):
        continue
    content = path.read_bytes()
    try:
        with wave.open(io.BytesIO(content)) as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16_000, 1, 2):
                continue
            pcm = audio.readframes(audio.getnframes())
    except (wave.Error, EOFError):
        if content[8:12] == b"WAVE":
            continue
        pcm = content
    if len(pcm) != 32_000:
        continue
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    if np.array_equal(inputs[0], samples):
        speech_files.append((path, samples))
assert len(speech_files) == 1, "preparation must read speech data from the installed package"
speech, samples = speech_files[0]
np.testing.assert_array_equal(inputs[0], samples)
notes = []
for path in speech.parent.iterdir():
    if (
        path == speech
        or not path.is_file()
        or path.suffix in {".py", ".pyc"}
        or "__pycache__" in path.relative_to(package).parts
        or any(part.endswith(".dist-info") for part in path.relative_to(package).parts)
    ):
        continue
    try:
        notes.append(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        continue
required = (
    "LibriSpeech", "SLR12", "libri1", "CC BY 4.0",
    "https://creativecommons.org/licenses/by/4.0/",
)
matching = [note for note in notes if all(token in note for token in required)]
assert matching, "speech data must have adjacent source and license attribution"
assert any(
    "crop" in note.lower()
    and re.search(r"16\s*(?:khz|,?000\s*hz)", note.lower())
    and "mono" in note.lower()
    and ("s16le" in note.lower() or "16-bit little-endian" in note.lower())
    and ("resampl" in note.lower() or "conver" in note.lower())
    for note in matching
), "attribution must describe the crop and 16 kHz mono s16le conversion"
assert engine.runtime_status() is not None
assert engine.health().state.value == "ready"
assert set(model_directory.rglob("*")) == assets_before, "preparation acquired a model asset"
"""
    try:
        _run(
            str(python),
            "-I",
            "-c",
            script,
            str(REPOSITORY_ROOT),
            str(model_directory),
            cwd=tmp_path,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr)


@pytest.mark.req("SP-LFFASTERWHISPER-002")
def test_namespace_exports_and_source_use_only_owned_public_surfaces() -> None:
    """The module exports its public API without importing private core surfaces."""

    module = importlib.import_module("listenflow.faster_whisper")

    assert set(module.__all__) == MODULE_EXPORTS
    assert all(getattr(module, name) is not None for name in MODULE_EXPORTS)

    source = REPOSITORY_ROOT / "src/listenflow/faster_whisper/__init__.py"
    import_nodes = [
        node
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    version_imports = {node.module for node in import_nodes if node.module.endswith("._version")}
    assert version_imports == {"listenflow.faster_whisper._version"}
    private_core_imports = {
        (node.module, alias.name)
        for node in import_nodes
        if node.module.startswith("listenflow")
        and node.module != "listenflow.faster_whisper._version"
        for alias in node.names
        if alias.name.startswith("_")
    }
    assert not private_core_imports


@pytest.mark.req("SP-LFFASTERWHISPER-012")
def test_core_lock_upgrade_advances_only_the_core_registry_version(tmp_path: Path) -> None:
    """A registry core upgrade selects a newer release and preserves other locked packages."""

    index = tmp_path / "simple"
    core_checkout = tmp_path / "listenflow"
    dependency_checkout = tmp_path / "idna"
    module_checkout = tmp_path / "module"
    _git("init", "--initial-branch=main", str(core_checkout), cwd=tmp_path)
    _git("init", "--initial-branch=main", str(dependency_checkout), cwd=tmp_path)
    _publish_fixture_version(core_checkout, index, "listenflow", "0.17.0")
    _publish_fixture_version(dependency_checkout, index, "idna", "3.19")
    _write_fixture_module(module_checkout)

    _run("uv", "lock", "--default-index", index.as_uri(), cwd=module_checkout)
    first_lock = _read_toml(module_checkout / "uv.lock")
    first_core = _assert_registry_locked_core(
        first_lock,
        registry_url=str(index),
    )
    assert first_core["version"] == "0.17.0"
    assert _locked_package(first_lock, "idna")["version"] == "3.19"

    _publish_fixture_version(core_checkout, index, "listenflow", "0.17.1")
    _publish_fixture_version(dependency_checkout, index, "idna", "3.20")
    _run(
        "uv",
        "lock",
        "--default-index",
        index.as_uri(),
        "--upgrade-package",
        "listenflow",
        cwd=module_checkout,
    )
    _run(
        "uv",
        "lock",
        "--default-index",
        index.as_uri(),
        "--check",
        cwd=module_checkout,
    )
    second_lock = _read_toml(module_checkout / "uv.lock")
    second_core = _assert_registry_locked_core(
        second_lock,
        registry_url=str(index),
    )

    assert second_core["version"] == "0.17.1"
    assert [package for package in first_lock["package"] if package["name"] != "listenflow"] == [
        package for package in second_lock["package"] if package["name"] != "listenflow"
    ]


@pytest.mark.req("SP-LFFASTERWHISPER-012")
def test_consumer_can_pin_a_core_commit_with_the_module_as_a_git_source(
    tmp_path: Path,
) -> None:
    """A consumer installing the module from Git can choose a compatible core commit."""

    core_remote = tmp_path / "listenflow.git"
    core_checkout = tmp_path / "listenflow"
    module_remote = tmp_path / "module"
    consumer = tmp_path / "consumer"
    _git("init", "--bare", "--initial-branch=main", str(core_remote), cwd=tmp_path)
    _git("init", "--initial-branch=main", str(core_checkout), cwd=tmp_path)
    _commit_fixture_version(core_checkout, "listenflow", "0.17.0")
    _git("remote", "add", "origin", core_remote.as_uri(), cwd=core_checkout)
    _git("push", "--quiet", "--set-upstream", "origin", "main", cwd=core_checkout)
    pinned_commit = _commit_fixture_version(core_checkout, "listenflow", "0.17.1")
    _git("push", "--quiet", cwd=core_checkout)
    _commit_module_copy(module_remote)
    _write_consumer(
        consumer,
        core_remote=core_remote.as_uri(),
        core_revision=pinned_commit,
        module_remote=module_remote.as_uri(),
    )

    _run("uv", "lock", cwd=consumer)

    lock = _read_toml(consumer / "uv.lock")
    repository, revision, commit = _git_identity(
        _locked_package(lock, "listenflow")["source"]["git"]
    )
    assert repository == core_remote.as_uri()
    assert revision == pinned_commit
    assert commit == pinned_commit
    assert _locked_package(lock, "listenflow")["version"] == "0.17.1"
    module = _locked_package(lock, "listenflow-faster-whisper")
    assert module["source"]["git"].startswith(module_remote.as_uri())
    assert {dependency["name"] for dependency in module["dependencies"]} == {
        "faster-whisper",
        "listenflow",
    }
