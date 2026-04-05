"""Pytest for reasoning llm_client: one test per method."""
import os
import pytest
from unittest.mock import patch, MagicMock
from src.reasoning.llm_client import LLMClient


def test_llm_client_init_defaults(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = LLMClient()
    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client.api_key is None


def test_llm_client_init_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    client = LLMClient()
    assert client.api_key == "sk-test"
    assert client.model == "gpt-4o-mini"


def test_llm_client_generate_reasoning_calls_openai(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch("src.reasoning.llm_client.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"action": "preserve", "rationale": "OK"}'}}]
        }
        mock_post.return_value.raise_for_status = MagicMock()
        client = LLMClient()
        result = client.generate_reasoning("Hello")
        assert result["action"] == "preserve"
        assert result["rationale"] == "OK"
        mock_post.assert_called_once()


def test_llm_client_generate_reasoning_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    client = LLMClient()
    client.provider = "unsupported"
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        client.generate_reasoning("Hi")


def test_llm_client_generate_reasoning_returns_fallback_on_request_error(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch("src.reasoning.llm_client.requests.post") as mock_post:
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        client = LLMClient()
        result = client.generate_reasoning("Hi")
        assert "action" in result
        assert result.get("action") == "preserve"


def test_llm_client_generate_reasoning_returns_fallback_on_invalid_json(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch("src.reasoning.llm_client.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "not valid json {"}}]
        }
        client = LLMClient()
        result = client.generate_reasoning("Hi")
        assert "action" in result or "error" in result
