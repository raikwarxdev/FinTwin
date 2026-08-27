import os, httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

keys = {
    "KEY_1": os.getenv("GROQ_API_KEY_1"),
    "KEY_2": os.getenv("GROQ_API_KEY_2"),
    "KEY_3": os.getenv("GROQ_API_KEY_3"),
    "KEY_fallback": os.getenv("GROQ_API_KEY"),
}

for name, key in keys.items():
    if not key:
        print(f"{name}: not found in .env")
        continue
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        h = resp.headers
        print(f"{name}:")
        print(f"  requests remaining/day : {h.get('x-ratelimit-remaining-requests')} / {h.get('x-ratelimit-limit-requests')}")
        print(f"  tokens   remaining/day : {h.get('x-ratelimit-remaining-tokens')} / {h.get('x-ratelimit-limit-tokens')}")
        print(f"  resets in (requests)   : {h.get('x-ratelimit-reset-requests')}")
        print(f"  resets in (tokens)     : {h.get('x-ratelimit-reset-tokens')}")
    except Exception as e:
        print(f"{name}: ERROR — {e}")
