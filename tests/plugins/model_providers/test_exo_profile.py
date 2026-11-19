"""Contract tests for the local Prosper EXO provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def exo_profile():
    import providers

    profile = providers.get_provider_profile("exo")
    assert profile is not None, "exo provider profile must be registered"
    return profile


def test_exo_profile_uses_local_openai_contract(exo_profile):
    assert exo_profile.base_url == "http://127.0.0.1:52415/v1"
    assert exo_profile.env_vars == ("EXO_API_KEY", "EXO_BASE_URL")
    assert exo_profile.default_max_tokens == 4096
    assert exo_profile.default_aux_model in exo_profile.fallback_models
    assert {"prosper-exo", "local-burtha"} <= set(exo_profile.aliases)


def test_exo_does_not_emit_ollama_reasoning_fields(exo_profile):
    extra_body, top_level = exo_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False, "effort": "none"}
    )
    assert extra_body == {}
    assert top_level == {}
