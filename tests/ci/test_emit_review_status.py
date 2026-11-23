"""Tests for scripts/ci/emit_review_status.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "emit_review_status.py"
_spec = importlib.util.spec_from_file_location("emit_review_status", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load emit_review_status.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["emit_review_status"] = _mod
_spec.loader.exec_module(_mod)


def _diff_anchor(path: str) -> str:
    """GitHub compare-page fragment for a path (sha256 hex of the path)."""
    return f"diff-{hashlib.sha256(path.encode()).hexdigest()}"


def test_ci_review_status_links_to_each_sensitive_file_change():
    files = [".github/workflows/ci.yaml", "apps/desktop/eslint.config.mjs"]
    results = _mod.build_results(
        ci_review=True,
        mcp_catalog=False,
        supply_chain=False,
        label_present=False,
        ci_review_files=json.dumps(files),
        repo_url="https://github.com/nousresearch/hermes-agent",
        base_sha="base456",
        head_sha="abc123",
    )

    expected_lines = ["**Sensitive files changed:**"]
    for path in files:
        expected_lines.append(
            f"- [`{path}`](https://github.com/nousresearch/hermes-agent/"
            f"compare/base456...abc123#{_diff_anchor(path)})"
        )
    assert results[0]["detail"] == "\n".join(expected_lines)


def test_approved_ci_review_is_visible_info():
    path = ".github/workflows/ci.yaml"
    results = _mod.build_results(
        ci_review=True,
        mcp_catalog=False,
        supply_chain=False,
        label_present=True,
        ci_review_files=f'["{path}"]',
        repo_url="https://github.com/nousresearch/hermes-agent",
        base_sha="base456",
        head_sha="abc123",
    )

    assert results == [{
        "kind": "info",
        "title": "CI-sensitive file review",
        "summary": (
            "PR touches sensitive files, but the `ci-reviewed` label has been "
            "added, approving them."
        ),
        "detail": (
            "**Sensitive files changed:**\n"
            f"- [`{path}`](https://github.com/nousresearch/hermes-agent/"
            f"compare/base456...abc123#{_diff_anchor(path)})"
        ),
    }]
