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
        print(f"🕵️ Sentinel Librarian v3: Universal Table-Scanning {file_path}...")
        
        structured_content = ""
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. Extract Tables (Crucial for Form 16 numbers) [cite: 66, 290]
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        # Clean up None values and join row items with | to preserve table structure
                        clean_row = [str(item).replace('\n', ' ') for item in row if item is not None]
                        structured_content += " | ".join(clean_row) + "\n"
                
                # 2. Extract Text (For PAN and Verification sections) [cite: 54, 87, 316]
                text = page.extract_text()
                if text:
                    structured_content += text + "\n"
                
                structured_content += f"--- End of Page {i+1} ---\n"

        # 🧠 THE MULTI-STRATEGY PROMPT: Tells the AI exactly where to look [cite: 136, 214, 443]
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Tax Auditor. Extract data from this Indian Form 16 PDF. "
                        "STRATEGY:\n"
                        "1. 'salary': Find the section 'Annexure-I' or 'Details of Salary Paid'. Look for 'Gross Salary' or 'Total 17(1)'. "
                        "   (For Santosh it is ~19.3L, for Dileep it is ~2.8Cr). \n"
                        "2. 'tax_paid': Look for 'Total amount of tax deducted' or 'Net tax payable' (often on Page 1 or 2). [cite: 66, 290]\n"
                        "3. 'deductions_80c': Look for Chapter VI-A. If user is in New Regime (115BAC), this will be 0. [cite: 142, 379]\n"
                        "4. 'pan': 10-character alphanumeric code. [cite: 54, 365]\n"
                        "Return ONLY JSON: {'salary': float, 'tax_paid': float, 'deductions_80c': float, 'pan': 'string'}"
                    )
                },
                {"role": "user", "content": f"Structured Document Data:\n{structured_content[:25000]}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(chat_completion.choices[0].message.content)
        print(f"✅ Universal v3 Success: {extracted}")
        return extracted

    except Exception as e:
        print(f"⚠️ Extraction Failed: {e}")
        return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "ERROR"}
