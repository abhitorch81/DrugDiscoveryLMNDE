from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="lemonade"
)

resp = client.chat.completions.create(
    model="Gemma-3-4b-it-GGUF",
    messages=[{"role": "user", "content": "Explain aspirin in 3 lines."}],
    temperature=0.2,
)

print(resp.choices[0].message.content)