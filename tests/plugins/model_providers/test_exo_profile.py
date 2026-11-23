"""Contract tests for the Burtha local provider (EXO replacement)."""

from __future__ import annotations

import json
import urllib.request

import pytest


@pytest.fixture
def exo_profile():
    import providers

    profile = providers.get_provider_profile("exo")
    assert profile is not None, "exo provider profile must be registered"
    return profile


def test_exo_profile_uses_burtha_local_contract(exo_profile):
    assert exo_profile.base_url == "http://127.0.0.1:18765/v1"
    assert exo_profile.env_vars == (
        "BURTHA_API_KEY",
        "EXO_API_KEY",
        "BURTHA_BASE_URL",
        "EXO_BASE_URL",
    )
    assert exo_profile.default_max_tokens == 4096
    assert exo_profile.default_aux_model == "burtha-local"
    assert exo_profile.default_aux_model in exo_profile.fallback_models
    assert {"prosper-exo", "local-burtha", "burtha-local", "burtha"} <= set(
        exo_profile.aliases
    )


def test_exo_does_not_emit_ollama_reasoning_fields(exo_profile):
    extra_body, top_level = exo_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False, "effort": "none"}
    )
    assert extra_body == {}
    assert top_level == {}


def test_burtha_local_llm_serves_openai_compatible_chat():
    from hermes_cli.burtha_local_llm import DEFAULT_MODEL, serve

    server = serve("127.0.0.1", 0, verbose=False)
    host, port = server.server_address
    base = f"http://{host}:{port}/v1"
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=2) as resp:
            models = json.loads(resp.read().decode("utf-8"))
        assert models["data"][0]["id"] == DEFAULT_MODEL

        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(
                {
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "ping from doctor"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            completion = json.loads(resp.read().decode("utf-8"))
        content = completion["choices"][0]["message"]["content"]
        assert "Burtha local endpoint online" in content
        assert "ping from doctor" in content
    finally:
        server.shutdown()
        server.server_close()


def test_burtha_credentials_accept_legacy_exo_base_url(monkeypatch):
    from hermes_cli.auth import (
        BURTHA_NOAUTH_PLACEHOLDER,
        resolve_api_key_provider_credentials,
    )

    monkeypatch.delenv("BURTHA_BASE_URL", raising=False)
    monkeypatch.delenv("BURTHA_API_KEY", raising=False)
    monkeypatch.delenv("EXO_API_KEY", raising=False)
    monkeypatch.setenv("EXO_BASE_URL", "http://127.0.0.1:52415/v1")

    creds = resolve_api_key_provider_credentials("exo")
    assert creds["base_url"] == "http://127.0.0.1:52415/v1"
    assert creds["api_key"] == BURTHA_NOAUTH_PLACEHOLDER


def test_burtha_local_does_not_hijack_auto_auxiliary(monkeypatch):
    """No-auth Burtha/EXO must not win the auto aux chain (like LM Studio)."""
    from unittest.mock import patch

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BURTHA_API_KEY", raising=False)
    monkeypatch.delenv("BURTHA_BASE_URL", raising=False)
    monkeypatch.delenv("EXO_API_KEY", raising=False)
    monkeypatch.delenv("EXO_BASE_URL", raising=False)

    from agent.auxiliary_client import get_text_auxiliary_client

    with patch("agent.auxiliary_client._read_nous_auth", return_value=None), patch(
        "agent.auxiliary_client._read_codex_access_token", return_value="codex-tok"
    ), patch("agent.auxiliary_client.OpenAI") as mock_openai:
        client, model = get_text_auxiliary_client()

    assert client is None
    assert model is None
    mock_openai.assert_not_called()
