import requests, json

url = "http://localhost:11434/api/generate"

while True:
    question = input("\nAsk something (or type 'exit' to quit): ").strip()

    if question.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not question:
        continue

    payload = {"model": "llama3.2", "prompt": question}

    try:
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    print(chunk.get("response", ""), end="", flush=True)
                    if chunk.get("done"):
                        print()
                        break
    except requests.exceptions.ConnectionError:
        print("Can't reach Ollama. Start it with: brew services start ollama")