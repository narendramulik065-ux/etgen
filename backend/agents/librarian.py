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
    client = Groq(api_key=api_key)

    try:
        print(f"🕵️ Sentinel Librarian: Scanning {file_path} for universal tax patterns...")
        
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            # We scan the entire document to find Annexures which might be at the end
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        # 🧠 THE UNIVERSAL PROMPT: Focused on legal markers rather than page numbers
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Indian Tax Auditor. Extract data from any Form 16 PDF. "
                        "Follow these strictly: "
                        "1. 'salary': Find the 'Gross Salary' (Total of 17(1), 17(2), and 17(3)). "
                        "2. 'tax_paid': Find 'Total Tax Deducted' or 'Net Tax Payable'. "
                        "3. 'deductions_80c': Find 'Section 80C' under Chapter VI-A. If 0 or New Regime, return 0.0. "
                        "4. 'pan': Extract the 10-character alphanumeric Employee PAN. "
                        "Return ONLY valid JSON: {'salary': float, 'tax_paid': float, 'deductions_80c': float, 'pan': 'string'}"
                    )
                },
                {"role": "user", "content": f"Document Text:\n{full_text[:20000]}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(chat_completion.choices[0].message.content)
        print(f"✅ Universal Extraction Success: {extracted}")
        return extracted

    except Exception as e:
        print(f"⚠️ Extraction Failed: {e}")
        return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "ERROR"}
