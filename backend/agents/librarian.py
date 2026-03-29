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
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " \n "

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional Indian Tax Auditor. Extract data from this Form 16.\n"
                        "RULES:\n"
                        "1. Find Gross Salary or Total salary u/s 17(1)\n"
                        "2. Find Net tax payable OR Total tax deducted\n"
                        "3. Find Section 80C deduction (final deductible amount)\n"
                        "4. Extract PAN\n"
                        "IMPORTANT:\n"
                        "- Different formats exist, search entire document\n"
                        "- Do NOT miss values if present\n"
                        "Return JSON"
                    )
                },
                {"role": "user", "content": f"Full Document Text:\n{full_text}"}
            ],
            model="model="llama-3.1-8b-instant"", 
            temperature=0,
            response_format={"type": "json_object"}
        )

        # ✅ SAFE JSON PARSE (prevents crash)
        content = chat_completion.choices[0].message.content
        try:
            raw = json.loads(content)
        except:
            print("⚠️ JSON parse failed:", content)
            return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "PARSE_ERROR"}

        # 🔥 NORMALIZATION (FIXED)
        salary = (
            raw.get("salary")
            or raw.get("Gross Salary")
            or raw.get("Total 17(1)")
            or raw.get("Total Salary")
            or 0.0
        )

        tax_paid = (
            raw.get("tax_paid")
            or raw.get("Net tax payable")
            or raw.get("Total tax deducted")
            or raw.get("Tax")
            or 0.0
        )

        deductions_80c = (
            raw.get("deductions_80c")
            or raw.get("Section 80C")
            or raw.get("80C")
            or raw.get("Total deduction under section 80C")
            or raw.get("Total deduction under section 80C, 80CCC and 80CCD(1)")
            or raw.get("Deduction in respect of life insurance premia")
            or 0.0
        )

        pan = raw.get("pan") or raw.get("PAN") or "UNKNOWN"

        extracted = {
            "salary": float(salary),
            "tax_paid": float(tax_paid),
            "deductions_80c": float(deductions_80c),
            "pan": pan
        }

        print(f"✅ Normalized Extraction: {extracted}")
        return extracted

    except Exception as e:
        print(f"⚠️ Groq Error: {e}. Returning fallback.")
        return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "ERROR"}
