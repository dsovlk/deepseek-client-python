# DeepSeek Client for Python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/deepseek-client-python)](https://pypi.org/project/deepseek-client-python/)
[![Python Versions](https://img.shields.io/pypi/pyversions/deepseek-client-python)](https://pypi.org/project/deepseek-client-python/)

Python client for DeepSeek based on requests

## Features

- 🚀 **Text & Chat Completions**: Generate natural language responses
- ⚙️ **Parameter Control**: Adjust temperature, top_p, presence_penalty, etc.
- 🌊 **Streaming Support**: Real-time response handling
- 📦 **Model Management**: List available models and set defaults
- 🔒 **Error Handling**: Robust API error management

## Installation

```bash
pip install deepseek-client-python
export DEEPSEEK_API_KEY="your-api-key-here"
```

## API Parameters

| Parameter          | Type    | Default | Description                          |
|--------------------|---------|---------|--------------------------------------|
| `temperature`      | `float` | 0.7     | Creativity control (0.0-2.0)         |
| `max_tokens`       | `int`   | 1024    | Maximum response length              |
| `top_p`            | `float` | 1.0     | Nucleus sampling (0.0-1.0)           |
| `presence_penalty` | `float` | 0.0     | Repetition control (-2.0-2.0)        |
| `stream`           | `bool`  | False   | Enable real-time streaming           |

## Usage

```python
 # Example usage for deepseek-client-python
    client = DeepSeekClient()

    # Example completion
    response = client.generate(
        prompt="Explain quantum computing in simple terms",
        temperature=0.5,
        max_tokens=500,
    )
    print(response["choices"][0]["text"])

```