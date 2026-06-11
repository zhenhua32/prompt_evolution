"""LiteLLM provider 回归测试。"""

from __future__ import annotations

from prompt_evolution.providers.litellm_provider import LiteLLMProvider


def test_normalizes_openai_compatible_model_with_explicit_base() -> None:
    provider = LiteLLMProvider(
        model="xiaomi/mimo-v2.5",
        api_key="test-key",
        api_base="https://openrouter.ai/api/v1",
    )

    assert provider.model == "openai/xiaomi/mimo-v2.5"
    assert provider._api_base == "https://openrouter.ai/api/v1"
    assert provider._api_key == "test-key"


def test_normalizes_openai_compatible_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    provider = LiteLLMProvider(model="xiaomi/mimo-v2.5")

    assert provider.model == "openai/xiaomi/mimo-v2.5"
    assert provider._api_base == "https://openrouter.ai/api/v1"
    assert provider._api_key == "test-key"


def test_keeps_known_provider_prefix_unchanged() -> None:
    provider = LiteLLMProvider(
        model="openrouter/xiaomi/mimo-v2.5",
        api_key="test-key",
        api_base="https://openrouter.ai/api/v1",
    )

    assert provider.model == "openrouter/xiaomi/mimo-v2.5"
    assert provider._api_base == "https://openrouter.ai/api/v1"