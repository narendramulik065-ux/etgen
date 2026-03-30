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
        return _empty_result("NO_KEY")

    client = Groq(api_key=api_key)

    try:
        print(f"📂 Librarian: Deep-scanning {file_path} for tax data...")
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        if not full_text.strip():
            print("⚠️ PDF text extraction returned empty.")
            return _empty_result("EMPTY_PDF")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional Indian Tax Auditor specializing in Form 16 analysis.\n\n"
                        "Extract EXACTLY these fields from the Form 16 document:\n\n"
                        "1. taxable_salary: This is the FINAL TAXABLE salary AFTER all section 10 exemptions.\n"
                        "   - In Form 16 Part B, this is LINE 6: 'Income chargeable under the head Salaries'\n"
                        "   - Formula in Form 16: Line 3 + Line 1(e) - Line 5\n"
                        "   - DO NOT return Gross Salary (line 1d). DO NOT return total salary from Part A.\n"
                        "   - This is the actual income that is taxed after exemptions like HRA, LTA, etc.\n\n"
                        "2. gross_salary: Raw gross salary before any deductions (line 1d in Part B).\n\n"
                        "3. net_tax_payable: The NET tax payable (line 19 in Part B: 'Net tax payable').\n"
                        "   - NOT the TDS amount from Part A.\n\n"
                        "4. deductions_80c: The DEDUCTIBLE amount under section 80C (not gross, the final deductible column).\n\n"
                        "5. deductions_80d: The deductible amount under section 80D (health insurance).\n\n"
                        "6. deductions_80ccd1b: The deductible amount under section 80CCD(1B) — NPS extra ₹50,000 slot.\n\n"
                        "7. standard_deduction: Standard deduction under section 16(ia) — typically ₹50,000.\n\n"
                        "8. gross_total_income: Line 9 in Part B: 'Gross total income'.\n\n"
                        "9. total_taxable_income: Line 12 in Part B: 'Total taxable income' (after all VI-A deductions).\n\n"
                        "10. total_vi_a_deductions: Line 11 in Part B: 'Aggregate of deductible amount under Chapter VI-A'.\n\n"
                        "11. pan: Employee PAN number.\n\n"
                        "12. employee_name: Employee full name.\n\n"
                        "13. assessment_year: Assessment year (e.g. 2023-24).\n\n"
                        "Return ONLY a valid JSON object with these exact keys. "
                        "Use 0.0 for any field not found. Do not add markdown, backticks, or explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Form 16 Document Text:\n\n{full_text}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = chat_completion.choices[0].message.content

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed: {e}\nRaw content: {content}")
            return _empty_result("PARSE_ERROR")

        print("🔍 RAW LLM STRUCTURE:", json.dumps(raw, indent=2))

        # --- Extract with multiple fallback key names ---
        taxable_salary = _get_first(raw, [
            "taxable_salary", "taxable_income_salary", "income_chargeable_salaries",
            "Income chargeable under the head Salaries", "line_6"
        ], 0.0)

        gross_salary = _get_first(raw, [
            "gross_salary", "Gross Salary", "Gross_Salary", "total_salary_17_1",
            "salary_17_1", "Total 17(1)"
        ], 0.0)

        net_tax_payable = _get_first(raw, [
            "net_tax_payable", "Net tax payable", "Net_Tax_Payable",
            "tax_payable", "Tax", "net_tax"
        ], 0.0)

        deductions_80c = _get_first(raw, [
            "deductions_80c", "Section 80C", "80C", "deduction_80c",
            "Total deduction under section 80C",
            "Total deduction under section 80C, 80CCC and 80CCD(1)"
        ], 0.0)

        deductions_80d = _get_first(raw, [
            "deductions_80d", "Section 80D", "80D", "deduction_80d",
            "health_insurance_deduction"
        ], 0.0)

        deductions_80ccd1b = _get_first(raw, [
            "deductions_80ccd1b", "80CCD_1B", "80CCD(1B)",
            "nps_additional_deduction", "deduction_80ccd1b"
        ], 0.0)

        standard_deduction = _get_first(raw, [
            "standard_deduction", "Standard deduction", "16_ia", "deduction_16ia"
        ], 50000.0)

        gross_total_income = _get_first(raw, [
            "gross_total_income", "Gross total income", "line_9"
        ], 0.0)

        total_taxable_income = _get_first(raw, [
            "total_taxable_income", "Total taxable income", "line_12",
            "taxable_income", "net_taxable_income"
        ], 0.0)

        total_vi_a_deductions = _get_first(raw, [
            "total_vi_a_deductions", "Aggregate of deductible amount under Chapter VI-A",
            "chapter_via_total", "vi_a_total"
        ], 0.0)

        pan = raw.get("pan") or raw.get("PAN") or "UNKNOWN"
        employee_name = raw.get("employee_name") or raw.get("Employee Name") or "UNKNOWN"
        assessment_year = raw.get("assessment_year") or raw.get("Assessment Year") or "UNKNOWN"

        # --- Fallback: derive taxable_salary if LLM missed it ---
        # Form 16 Part B line 6 = line3 + line1e - line5
        # If LLM gives us gross_total_income and standard_deduction, we can compute
        if taxable_salary <= 0 and gross_total_income > 0:
            print("⚠️ taxable_salary missing — falling back to gross_total_income")
            taxable_salary = gross_total_income

        # Final safety: if taxable_salary still unreasonably large, use total_taxable_income
        # (e.g., LLM returned gross ₹2.84Cr instead of taxable ₹6.85L)
        if total_taxable_income > 0 and taxable_salary > total_taxable_income * 5:
            print(f"⚠️ taxable_salary {taxable_salary} looks like gross — using total_taxable_income {total_taxable_income}")
            taxable_salary = gross_total_income if gross_total_income > 0 else total_taxable_income

        extracted = {
            "taxable_salary": float(taxable_salary),
            "gross_salary": float(gross_salary),
            "tax_paid": float(net_tax_payable),
            "deductions_80c": float(deductions_80c),
            "deductions_80d": float(deductions_80d),
            "deductions_80ccd1b": float(deductions_80ccd1b),
            "standard_deduction": float(standard_deduction),
            "gross_total_income": float(gross_total_income),
            "total_taxable_income": float(total_taxable_income),
            "total_vi_a_deductions": float(total_vi_a_deductions),
            "pan": pan,
            "employee_name": employee_name,
            "assessment_year": assessment_year,
        }

        print(f"✅ Normalized Extraction: {json.dumps(extracted, indent=2)}")
        return extracted

    except Exception as e:
        print(f"⚠️ Groq Error: {e}. Returning fallback.")
        return _empty_result("ERROR")


def _get_first(d: dict, keys: list, default):
    """Return the first non-zero, non-None value found among the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None and v != 0 and v != "":
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return default


def _empty_result(reason: str) -> dict:
    return {
        "taxable_salary": 0.0,
        "gross_salary": 0.0,
        "tax_paid": 0.0,
        "deductions_80c": 0.0,
        "deductions_80d": 0.0,
        "deductions_80ccd1b": 0.0,
        "standard_deduction": 50000.0,
        "gross_total_income": 0.0,
        "total_taxable_income": 0.0,
        "total_vi_a_deductions": 0.0,
        "pan": reason,
        "employee_name": "UNKNOWN",
        "assessment_year": "UNKNOWN",
    }
