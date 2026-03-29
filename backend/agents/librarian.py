import os
import json
import pdfplumber
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

# Load Environment Variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

def get_tax_data(file_path: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ ERROR: GROQ_API_KEY not found.")
        return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "NO_KEY"}

    client = Groq(api_key=api_key)

    try:
        print(f"📂 Librarian: Deep-scanning {file_path} for universal data...")
        
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            # FIX: Scans every page to find Annexures regardless of page number 
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " \n "

        # 🎯 UNIVERSAL PROMPT: No specific names or numbers to avoid hallucination 
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional Indian Tax Auditor. Extract data from this Form 16. "
                        "RULES:\n"
                        "1. 'salary': Find 'Gross Salary' or 'Total 17(1)'. Do NOT assume or guess the range.\n"
                        "2. 'tax_paid': Find 'Net tax payable' or 'Total tax deducted'. If zero, return 0.0.\n"
                        "3. 'deductions_80c': Find Section 80C. If New Regime (115BAC), return 0.0.\n"
                        "4. 'pan': Extract the 10-character alphanumeric PAN.\n"
                        "Return ONLY valid JSON: {'salary': float, 'tax_paid': float, 'deductions_80c': float, 'pan': 'string'}"
                    )
                },
                {"role": "user", "content": f"Full Document Text:\n{full_text[:20000]}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(chat_completion.choices[0].message.content)
        print(f"✅ Universal Extraction Success: {extracted}")
        return extracted

    except Exception as e:
        print(f"⚠️ Groq Error: {e}. Returning fallback.")
        return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "ERROR"}
