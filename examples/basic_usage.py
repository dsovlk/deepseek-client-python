import logging
import os

logging.basicConfig(level=logging.DEBUG)

from deepseek_client.client import DeepSeekClient

if __name__ == "__main__":
    # Example usage for deepseek-client-python
    client = DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com/v1"
    )

    try:
        # Try chat completions instead of text completions
        response = client.chat(
            messages=[
                {"role": "user", "content": "Explain quantum computing in simple terms"}
            ],
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=100,
        )
        print("\nResponse:", response["choices"][0]["message"]["content"])

    except Exception as e:
        print(f"\nError: {str(e)}")
        if hasattr(e, "response"):
            print("Server response:", e.response.text)
