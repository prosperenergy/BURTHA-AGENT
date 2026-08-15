"""Prosper EXO local OpenAI-compatible provider."""

from providers import register_provider
from providers.base import ProviderProfile


exo = ProviderProfile(
    name="exo",
    aliases=("prosper-exo", "local-burtha"),
    display_name="Prosper EXO",
    description="Local EXO OpenAI-compatible inference endpoint.",
    env_vars=("EXO_API_KEY", "EXO_BASE_URL"),
    base_url="http://127.0.0.1:52415/v1",
    auth_type="api_key",
    fallback_models=("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",),
    default_max_tokens=4096,
    default_aux_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
)

register_provider(exo)
