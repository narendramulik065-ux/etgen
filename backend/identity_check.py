import os
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# Pathing
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY is not set. Please check your .env file.")
    exit(1)

genai.configure(api_key=api_key)

print("--- AVAILABLE MODELS FOR YOUR KEY ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # We want the string after 'models/'
            print(f"✅ USE THIS STRING: {m.name.replace('models/', '')}")
except Exception as e:
    print(f"❌ API Key Error: {e}")
