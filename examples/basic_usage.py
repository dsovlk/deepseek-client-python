if __name__ == "__main__":
    # Example usage for deepseek-client-python
    client = DeepSeekClient()

    # Example completion
    response = client.generate(
        prompt="Explain quantum computing in simple terms",
        temperature=0.5,
        max_tokens=500,
    )
    print(response["choices"][0]["text"])
