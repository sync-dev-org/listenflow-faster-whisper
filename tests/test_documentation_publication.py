from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPOSITORY_ROOT / ".github/workflows/docs.yml"


def _workflow() -> dict[str, Any]:
    if not WORKFLOW.is_file() and not (REPOSITORY_ROOT / "AGENTS.md").is_file():
        pytest.skip("the source distribution does not include GitHub configuration")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.mark.req("SP-LFFASTERWHISPER-021")
def test_public_snapshot_includes_the_documentation_workflow_without_issue_templates() -> None:
    """The public snapshot receives the site workflow and excludes private issue templates."""

    configuration = REPOSITORY_ROOT / "noren.yaml"
    if not configuration.is_file() and not (REPOSITORY_ROOT / "AGENTS.md").is_file():
        pytest.skip("the public snapshot does not include development publication settings")
    publication = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    workflow_path = "/" + WORKFLOW.relative_to(REPOSITORY_ROOT).as_posix()

    assert WORKFLOW.is_file()
    assert workflow_path in publication["include"].splitlines()
    assert not any(workflow_path.startswith(path) for path in publication["exclude"].splitlines())
    assert "/.github/ISSUE_TEMPLATE/" in publication["exclude"].splitlines()


@pytest.mark.req("SP-LFFASTERWHISPER-021")
def test_public_main_and_manual_runs_build_strict_documentation_and_deploy_pages() -> None:
    """Public main pushes and manual runs build strict documentation and deploy its artifact."""

    workflow = _workflow()
    assert workflow["on"] == {"push": {"branches": ["main"]}, "workflow_dispatch": None}
    assert workflow["concurrency"] == {"group": "pages", "cancel-in-progress": False}
    (job,) = workflow["jobs"].values()
    assert job["if"] == "github.repository == 'sync-dev-org/listenflow-faster-whisper'"
    assert job["environment"] == {
        "name": "github-pages",
        "url": "${{ steps.deployment.outputs.page_url }}",
    }
    steps = job["steps"]
    actions = {step["uses"].split("@")[0]: step for step in steps if "uses" in step}
    assert "actions/checkout" in actions
    assert "astral-sh/setup-uv" in actions
    commands = [shlex.split(step["run"]) for step in steps if "run" in step]
    assert ["uv", "sync", "--locked", "--only-group", "docs"] in commands
    assert ["uv", "run", "--no-sync", "mkdocs", "build", "--strict"] in commands
    assert actions["actions/upload-pages-artifact"]["with"]["path"] == "site"
    assert actions["actions/deploy-pages"]["id"] == "deployment"
    build_index = next(index for index, step in enumerate(steps) if "mkdocs" in step.get("run", ""))
    assert build_index < steps.index(actions["actions/upload-pages-artifact"])
    assert steps.index(actions["actions/upload-pages-artifact"]) < steps.index(
        actions["actions/deploy-pages"]
    )


@pytest.mark.req("SP-LFFASTERWHISPER-022")
def test_actions_only_build_and_publish_documentation_with_limited_permissions() -> None:
    """Actions only build and publish the site with the three Pages permissions and no secrets."""

    workflow = _workflow()
    assert workflow["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert re.search(r"\bsecrets\s*[.\[]", WORKFLOW.read_text(encoding="utf-8")) is None
    allowed_actions = {
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/configure-pages",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    }
    allowed_commands = {
        ("uv", "sync", "--locked", "--only-group", "docs"),
        ("uv", "run", "--no-sync", "mkdocs", "build", "--strict"),
    }
    for job in workflow["jobs"].values():
        assert "permissions" not in job
        assert "uses" not in job
        for step in job["steps"]:
            if "uses" in step:
                action, version = step["uses"].split("@")
                assert action in allowed_actions
                assert re.fullmatch(r"v\d+(?:\.\d+)*|[0-9a-f]{40}", version)
            if "run" in step:
                assert tuple(shlex.split(step["run"])) in allowed_commands


@pytest.mark.req("SP-LFFASTERWHISPER-021")
def test_documentation_dependencies_include_core_and_remain_available_to_developers() -> None:
    """The docs group includes the API base classes and is included in the developer environment."""

    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]
    requirements = [Requirement(value) for value in groups["docs"]]

    assert {requirement.name for requirement in requirements} == {
        "listenflow",
        "mkdocs-material",
        "mkdocstrings",
    }
    assert all(requirement.url is None for requirement in requirements)
    assert {"include-group": "docs"} in groups["dev"]
    assert not any(
        isinstance(value, str) and Requirement(value).name in {"mkdocs-material", "mkdocstrings"}
        for value in groups["dev"]
    )
