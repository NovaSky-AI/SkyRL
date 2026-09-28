import json
import os
import urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
key = os.environ["OPENAI_API_KEY"]

def chat(messages):
    body = json.dumps({"model": "gpt-4o-mini", "messages": messages, "max_tokens": 64}).encode()
    request = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]

history = [{"role": "system", "content": "You are a terse assistant."}]
for prompt in ["list the files", "now read config.yaml", "summarize what you found"]:
    history.append({"role": "user", "content": prompt})
    reply = chat(history)
    history.append({"role": "assistant", "content": reply})
    print(reply)