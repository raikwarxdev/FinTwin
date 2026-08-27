"""Check quota status for all 3 Groq keys."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

keys = {
    "KEY_1": os.getenv("GROQ_API_KEY_1"),
    "KEY_2": os.getenv("GROQ_API_KEY_2"),
    "KEY_3": os.getenv("GROQ_API_KEY_3"),
    "KEY_fallback": os.getenv("GROQ_API_KEY"),
}

any_working = False
for name, key in keys.items():
    if not key:
        print(f"{name}: not found in .env")
        continue
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    try:
        r = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print(f"{name}: SUCCESS — {r.choices[0].message.content}")
        any_working = True
    except Exception as e:
        if "429" in str(e):
            print(f"{name}: QUOTA EXHAUSTED")
        else:
            print(f"{name}: ERROR — {e}")

if any_working:
    print("\n✅ At least one key is working — ready to run")
else:
    print("\n❌ All keys exhausted — wait for reset at 5:30 AM IST")
