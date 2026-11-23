"""Burtha local OpenAI-compatible provider (EXO replacement).

Prosper originally pointed this profile at a local EXO cluster
(``127.0.0.1:52415``, MLX). That endpoint is unavailable in Cursor Cloud,
Linux CI, and most non-Mac environments. The default now targets the
controllable Burtha local server:

    python -m hermes_cli.burtha_local_llm

Mac/EXO users can still override with ``BURTHA_BASE_URL`` or legacy
``EXO_BASE_URL`` pointing at ``http://127.0.0.1:52415/v1``.
"""

from providers import register_provider
from providers.base import ProviderProfile


exo = ProviderProfile(
    name="exo",
    aliases=("prosper-exo", "local-burtha", "burtha-local", "burtha"),
    display_name="Burtha Local",
    description=(
        "Controllable local OpenAI-compatible endpoint for Burtha "
        "(replaces EXO when the Apple-Silicon cluster is unavailable)."
    ),
    env_vars=(
        "BURTHA_API_KEY",
        "EXO_API_KEY",
        "BURTHA_BASE_URL",
        "EXO_BASE_URL",
    ),
    base_url="http://127.0.0.1:18765/v1",
    auth_type="api_key",
    fallback_models=("burtha-local",),
    default_max_tokens=4096,
    default_aux_model="burtha-local",
)

register_provider(exo)
