"""
DeepSeek Client for Python

Official Python client for interacting with DeepSeek's LLM API.
Repository: https://github.com/dsvolk/deepseek-client-python
"""

import os
import requests
from typing import Optional, Dict, Any, List, Iterator


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        default_model: str = "deepseek-chat",
        default_temperature: float = 0.7,
        timeout: int = 30,
    ):
        """
        Initialize the DeepSeek API client

        :param api_key: API key (default: DEEPSEEK_API_KEY environment variable)
        :param base_url: Base API URL
        :param default_model: Default model for API requests
        :param default_temperature: Default sampling temperature (0.0-2.0)
        :param timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set DEEPSEEK_API_KEY environment variable "
                "or provide explicitly."
            )

        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "deepseek-client-python/1.0.0",
        }

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as err:
            try:
                error_data = response.json()
                message = error_data.get("message", "Unknown error")
                code = error_data.get("code", "unknown")
            except ValueError:
                message = response.text
                code = "parse_error"

            raise requests.exceptions.HTTPError(
                f"DeepSeek API Error {response.status_code} ({code}): {message}"
            ) from err

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text completion

        :param prompt: Input text/prompt
        :param model: Override default model
        :param temperature: Sampling temperature (0.0-2.0)
        :param max_tokens: Maximum tokens to generate
        :param top_p: Nucleus sampling threshold (0.0-1.0)
        :param presence_penalty: Repetition penalty (-2.0-2.0)
        :param stream: Enable streaming response
        :return: API response
        """
        url = f"{self.base_url}/completions"
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "stream": stream,
            **kwargs,
        }

        response = requests.post(
            url, headers=self.headers, json=payload, timeout=self.timeout, stream=stream
        )

        return self._handle_response(response) if not stream else response

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create chat completion

        :param messages: List of message dictionaries
        :param model: Override default model
        :param temperature: Sampling temperature (0.0-2.0)
        :param max_tokens: Maximum tokens to generate
        :param top_p: Nucleus sampling threshold (0.0-1.0)
        :param presence_penalty: Repetition penalty (-2.0-2.0)
        :param stream: Enable streaming response
        :return: API response
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "stream": stream,
            **kwargs,
        }

        response = requests.post(
            url, headers=self.headers, json=payload, timeout=self.timeout, stream=stream
        )

        return self._handle_response(response) if not stream else response

    def stream_response(self, response: requests.Response) -> Iterator[str]:
        """
        Handle streaming responses

        :param response: Streaming response object
        :yield: Incremental response chunks
        """
        try:
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")
        except requests.exceptions.ChunkedEncodingError as e:
            raise requests.exceptions.RequestException(f"Stream error: {str(e)}") from e

    def list_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of available models"""
        url = f"{self.base_url}/models"
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        return self._handle_response(response).get("data", [])

    def set_default_model(self, model: str) -> None:
        """Update default model for subsequent requests"""
        self.default_model = model

    def set_default_temperature(self, temperature: float) -> None:
        """Update default temperature (0.0-2.0)"""
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.default_temperature = temperature
