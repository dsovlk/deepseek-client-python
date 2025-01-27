import pytest
import requests
from unittest.mock import Mock, patch
from deepseek_client.client import DeepSeekClient


def test_client_initialization():
    client = DeepSeekClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.base_url == "https://api.deepseek.com/v1"
    assert client.default_model == "deepseek-chat"
    assert client.headers["Authorization"] == "Bearer test_key"


def test_missing_api_key():
    with pytest.raises(ValueError) as excinfo:
        DeepSeekClient(api_key=None)
    assert "API key required" in str(excinfo.value)


@patch("requests.post")
def test_generate_success(mock_post, mock_client):
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"text": "Test response"}],
        "usage": {"total_tokens": 10},
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    response = mock_client.generate(prompt="Test prompt")
    assert response["choices"][0]["text"] == "Test response"
    mock_post.assert_called_once_with(
        "https://mock.api.deepseek.com/completions",
        headers=mock_client.headers,
        json={
            "model": "deepseek-chat",
            "prompt": "Test prompt",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "stream": False,
        },
        timeout=30,
        stream=False,
    )


@patch("requests.post")
def test_chat_streaming(mock_post, mock_client):
    mock_response = Mock()
    # Simulate SSE format with "data: " prefix
    mock_response.iter_lines.return_value = [
        b'data: {"choices": [{"delta": {"content": "Chunk1"}}]}',
        b'data: {"choices": [{"delta": {"content": "Chunk2"}}]}',
    ]
    mock_post.return_value = mock_response

    stream = mock_client.chat(
        messages=[{"role": "user", "content": "Test"}], stream=True
    )
    chunks = list(mock_client.stream_response(stream))

    # Include "data: " prefix in expected values
    assert chunks == [
        'data: {"choices": [{"delta": {"content": "Chunk1"}}]}',
        'data: {"choices": [{"delta": {"content": "Chunk2"}}]}',
    ]


def test_parameter_validation(mock_client):
    with pytest.raises(ValueError) as excinfo:
        mock_client.set_default_temperature(2.1)
    assert "Temperature must be between 0.0 and 2.0" in str(excinfo.value)


import pytest
import requests
from unittest.mock import Mock, patch
from deepseek_client.client import DeepSeekClient


@patch("requests.post")
def test_api_error_handling(mock_post):
    # Setup mock client
    client = DeepSeekClient(
        api_key="test_key", base_url="https://mock.api.deepseek.com"
    )

    # Configure mock response
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "message": "Invalid request parameters",
        "code": "invalid_request",
    }
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "400 Client Error", response=mock_response
    )
    mock_post.return_value = mock_response

    # Test error handling
    with pytest.raises(requests.exceptions.HTTPError) as excinfo:
        client.generate(prompt="Test")

    # Verify error message contents
    assert "400" in str(excinfo.value)
    assert "Invalid request parameters" in str(excinfo.value)
    assert "invalid_request" in str(excinfo.value)


@patch("requests.get")
def test_list_models(mock_get, mock_client):
    mock_response = Mock()
    mock_response.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
    mock_get.return_value = mock_response

    models = mock_client.list_models()
    assert len(models) == 2
    mock_get.assert_called_once_with(
        "https://mock.api.deepseek.com/models", headers=mock_client.headers, timeout=30
    )
