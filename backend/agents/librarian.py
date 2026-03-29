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
        print(f"🕵️ Sentinel Librarian v4: True Universal Scan for {file_path}...")
        
        structured_content = ""
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # 1. Extract Tables to preserve numeric integrity
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        clean_row = [str(item).replace('\n', ' ') for item in row if item is not None]
                        structured_content += " | ".join(clean_row) + "\n"
                
                # 2. Extract Text for general context
                text = page.extract_text()
                if text:
                    structured_content += text + "\n"
                
                structured_content += f"--- End of Page {i+1} ---\n"

        # 🧠 THE HINT-FREE PROMPT: Strictly data-driven
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous Indian Tax Auditor. Extract data from this Form 16. "
                        "RULES:\n"
                        "1. 'salary': Find the Gross Salary (Total 17(1)). Do NOT guess or scale this number.\n"
                        "2. 'tax_paid': Find 'Total amount of tax deducted' or 'Net tax payable'. If it says 'Zero' or '0.00', return 0.0.\n"
                        "3. 'deductions_80c': Extract the amount listed under Section 80C. \n"
                        "4. 'pan': Extract the 10-character alphanumeric PAN.\n"
                        "Return ONLY valid JSON: {'salary': float, 'tax_paid': float, 'deductions_80c': float, 'pan': 'string'}"
                    )
                },
                {"role": "user", "content": f"Document Data:\n{structured_content[:30000]}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        extracted = json.loads(chat_completion.choices[0].message.content)
        print(f"✅ True Universal Success: {extracted}")
        return extracted

    except Exception as e:
        print(f"⚠️ Extraction Failed: {e}")
        return {"salary": 0.0, "tax_paid": 0.0, "deductions_80c": 0.0, "pan": "ERROR"}
