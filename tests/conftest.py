import pytest
from deepseek_client.client import DeepSeekClient


@pytest.fixture
def mock_client():
    return DeepSeekClient(api_key="test_key", base_url="https://mock.api.deepseek.com")
