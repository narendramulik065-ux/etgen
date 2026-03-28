import os
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# Load project .env
env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env")
    exit(1)

print(f"📡 Testing Groq with key: {api_key[:10]}...")

try:
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say 'Groq is Live' if you can hear me.",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    print(f"✅ Success! Response: {chat_completion.choices[0].message.content}")
except Exception as e:
    print(f"❌ Groq Test Failed: {e}")
