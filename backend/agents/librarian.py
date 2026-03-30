import os
import json
import pdfplumber
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from project root .env
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_tax_data(file_path: str) -> dict:
    """
    Extract all tax-relevant fields from a Form 16 PDF.
    Returns a fully-populated dict. Never raises — always returns a
    fallback dict on any error so callers don't need to guard.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not found in environment.")
        return _empty_result("NO_KEY")

    client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Step 1 — Extract all text from the PDF
    # ------------------------------------------------------------------
    try:
        print(f"Librarian: scanning {file_path} ...")
        full_text = _extract_pdf_text(file_path)
    except Exception as e:
        print(f"WARNING: PDF read error: {e}")
        return _empty_result("PDF_READ_ERROR")

    if not full_text.strip():
        print("WARNING: PDF text extraction returned empty — possibly scanned/image PDF.")
        return _empty_result("EMPTY_PDF")

    # ------------------------------------------------------------------
    # Step 2 — Ask Groq LLM to extract structured fields
    # ------------------------------------------------------------------
    system_prompt = """You are a professional Indian Tax Auditor specialising in Form 16 (AY 2023-24 and later).

Extract EXACTLY the following fields from the Form 16 text. Return ONLY a valid JSON object.
Use 0.0 for any numeric field not found. Use empty string "" for text fields not found.
Do NOT add markdown fences, backticks, or any explanation — raw JSON only.

FIELD DEFINITIONS (read carefully):

1. "taxable_salary"
   Form 16 Part B, LINE 6: "Income chargeable under the head Salaries"
   Formula: Line 3 (salary from current employer) + Line 1(e) (other employer salary) minus Line 5 (Sec 16 deductions)
   THIS IS NOT Gross Salary (line 1d). It is typically a much smaller number.
   Example: if gross is 28412100 but line 6 shows 685304, return 685304.

2. "gross_salary"
   Form 16 Part B, LINE 1(d): total gross salary before any exemptions.

3. "other_employer_salary"
   Form 16 Part B, LINE 1(e): salary from previous or other employer(s).

4. "net_tax_payable"
   Form 16 Part B, LINE 19: "Net tax payable (17-18)". NOT the TDS from Part A.

5. "tax_on_total_income"
   Form 16 Part B, LINE 13: "Tax on total income".

6. "health_edu_cess"
   Form 16 Part B, LINE 16: "Health and education cess".

7. "rebate_87a"
   Form 16 Part B, LINE 14: "Rebate under section 87A".

8. "deductions_80c"
   Form 16 Part B, Section 10(d): Total deductible amount under 80C + 80CCC + 80CCD(1) combined.
   Use the DEDUCTIBLE AMOUNT column (right column), not gross amount.

9. "deductions_80ccd1b"
   Form 16 Part B, Section 10(e): Deductions under 80CCD(1B) — additional NPS slot, max 50000.

10. "deductions_80d"
    Form 16 Part B, Section 10(g): Deductible amount for health insurance under 80D.

11. "deductions_80e"
    Form 16 Part B, Section 10(h): Deductible amount for education loan interest under 80E.

12. "deductions_80g"
    Form 16 Part B, Section 10(i): Deductible amount for donations under 80G.

13. "deductions_80tta"
    Form 16 Part B, Section 10(j): Deductible amount under 80TTA savings account interest.

14. "standard_deduction"
    Form 16 Part B, Section 4(a): Standard deduction under 16(ia). Typically 50000.

15. "gross_total_income"
    Form 16 Part B, LINE 9: "Gross total income (6+8)".

16. "total_taxable_income"
    Form 16 Part B, LINE 12: "Total taxable income (9-11)".

17. "total_vi_a_deductions"
    Form 16 Part B, LINE 11: "Aggregate of deductible amount under Chapter VI-A".

18. "section_10_exemptions"
    Form 16 Part B, LINE 2(h): Total exemptions claimed under section 10 (HRA, LTA, treaty, etc.).

19. "pan"
    Employee PAN number (e.g., ABCDE1234F).

20. "employee_name"
    Employee full name.

21. "employer_name"
    Employer or company name.

22. "assessment_year"
    Assessment year string (e.g., "2023-24").

23. "period_from"
    Employment period start date as string (e.g., "01-Apr-2022").

24. "period_to"
    Employment period end date as string (e.g., "31-Mar-2023").

Return exactly these 24 keys. Numbers as floats, strings as strings."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Form 16 Document Text:\n\n{full_text}"},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_content = chat_completion.choices[0].message.content
    except Exception as e:
        print(f"WARNING: Groq API call failed: {e}")
        return _empty_result("GROQ_ERROR")

    # ------------------------------------------------------------------
    # Step 3 — Parse and normalise the LLM response
    # ------------------------------------------------------------------
    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"WARNING: JSON parse failed: {e}\nContent was: {raw_content[:500]}")
        return _empty_result("PARSE_ERROR")

    print("RAW LLM OUTPUT:", json.dumps(raw, indent=2))

    # Pull each field with multiple key fallbacks for LLM naming variations
    taxable_salary = _get_float(raw, [
        "taxable_salary", "income_chargeable_salaries",
        "Income chargeable under the head Salaries", "line_6",
        "taxable_income_salary",
    ], 0.0)

    gross_salary = _get_float(raw, [
        "gross_salary", "Gross Salary", "Gross_Salary",
        "gross_salary_17_1", "total_gross_salary",
    ], 0.0)

    other_employer_salary = _get_float(raw, [
        "other_employer_salary", "salary_other_employer",
        "reported_salary_other_employer", "line_1e",
    ], 0.0)

    net_tax_payable = _get_float(raw, [
        "net_tax_payable", "Net tax payable", "net_tax",
        "tax_payable_line19", "line_19",
    ], 0.0)

    tax_on_total_income = _get_float(raw, [
        "tax_on_total_income", "Tax on total income", "line_13",
        "tax_on_income",
    ], 0.0)

    health_edu_cess = _get_float(raw, [
        "health_edu_cess", "Health and education cess",
        "cess", "line_16", "education_cess",
    ], 0.0)

    rebate_87a = _get_float(raw, [
        "rebate_87a", "Rebate under section 87A",
        "rebate_87A", "section_87a_rebate",
    ], 0.0)

    deductions_80c = _get_float(raw, [
        "deductions_80c", "Section 80C", "80C",
        "Total deduction under section 80C, 80CCC and 80CCD(1)",
        "total_80c_80ccc_80ccd1", "deduction_80c",
    ], 0.0)

    deductions_80ccd1b = _get_float(raw, [
        "deductions_80ccd1b", "80CCD(1B)", "80CCD_1B",
        "nps_additional", "section_80ccd1b", "deduction_80ccd1b",
    ], 0.0)

    deductions_80d = _get_float(raw, [
        "deductions_80d", "Section 80D", "80D",
        "health_insurance_premium", "section_80d", "deduction_80d",
    ], 0.0)

    deductions_80e = _get_float(raw, [
        "deductions_80e", "80E", "education_loan_interest", "deduction_80e",
    ], 0.0)

    deductions_80g = _get_float(raw, [
        "deductions_80g", "80G", "donations_80g", "deduction_80g",
    ], 0.0)

    deductions_80tta = _get_float(raw, [
        "deductions_80tta", "80TTA", "savings_interest_80tta", "deduction_80tta",
    ], 0.0)

    standard_deduction = _get_float(raw, [
        "standard_deduction", "Standard deduction", "16_ia",
        "standard_deduction_16ia",
    ], 50000.0)
    # Safety: standard deduction is always at least 50000 under current rules
    if standard_deduction <= 0:
        standard_deduction = 50000.0

    gross_total_income = _get_float(raw, [
        "gross_total_income", "Gross total income", "line_9",
    ], 0.0)

    total_taxable_income = _get_float(raw, [
        "total_taxable_income", "Total taxable income", "line_12",
        "net_taxable_income",
    ], 0.0)

    total_vi_a_deductions = _get_float(raw, [
        "total_vi_a_deductions",
        "Aggregate of deductible amount under Chapter VI-A",
        "chapter_via_total", "line_11",
    ], 0.0)

    section_10_exemptions = _get_float(raw, [
        "section_10_exemptions", "Total amount of exemption claimed under section 10",
        "sec10_total", "line_2h",
    ], 0.0)

    pan             = _get_str(raw, ["pan", "PAN", "employee_pan"])
    employee_name   = _get_str(raw, ["employee_name", "Employee Name", "name"])
    employer_name   = _get_str(raw, ["employer_name", "Employer Name", "company_name"])
    assessment_year = _get_str(raw, ["assessment_year", "Assessment Year", "ay"])
    period_from     = _get_str(raw, ["period_from", "From", "employment_from"])
    period_to       = _get_str(raw, ["period_to", "To", "employment_to"])

    # ------------------------------------------------------------------
    # Step 4 — Sanity checks and derived fallbacks
    # ------------------------------------------------------------------

    # Detect if LLM accidentally returned gross salary as taxable_salary.
    if gross_total_income > 0 and taxable_salary > gross_total_income * 3:
        print(
            f"WARNING: taxable_salary ({taxable_salary:,.0f}) far exceeds "
            f"gross_total_income ({gross_total_income:,.0f}) — LLM likely returned gross. Overriding."
        )
        taxable_salary = gross_total_income

    # If taxable_salary is still 0, derive from gross_total_income
    if taxable_salary <= 0 and gross_total_income > 0:
        print("WARNING: taxable_salary is 0, falling back to gross_total_income.")
        taxable_salary = gross_total_income

    # Derive total_taxable_income if missing
    if total_taxable_income <= 0 and taxable_salary > 0:
        derived = max(taxable_salary - total_vi_a_deductions, 0.0)
        print(f"Derived total_taxable_income = {derived:,.0f}")
        total_taxable_income = derived

    # Derive gross_total_income if missing
    if gross_total_income <= 0 and taxable_salary > 0:
        gross_total_income = taxable_salary

    # Derive net_tax_payable if missing but sub-components available
    if net_tax_payable <= 0 and tax_on_total_income > 0:
        derived_tax = round(tax_on_total_income + health_edu_cess - rebate_87a, 2)
        print(f"Derived net_tax_payable = {derived_tax:,.2f}")
        net_tax_payable = derived_tax

    # Derive total_vi_a_deductions if missing but individual deductions present
    if total_vi_a_deductions <= 0:
        derived_via = (
            deductions_80c + deductions_80ccd1b + deductions_80d +
            deductions_80e + deductions_80g + deductions_80tta
        )
        if derived_via > 0:
            print(f"Derived total_vi_a_deductions = {derived_via:,.0f}")
            total_vi_a_deductions = derived_via

    # ------------------------------------------------------------------
    # Step 5 — Return the fully normalised dict
    # ------------------------------------------------------------------
    extracted = {
        # Core salary figures
        "taxable_salary":          float(taxable_salary),
        "gross_salary":            float(gross_salary),
        "other_employer_salary":   float(other_employer_salary),
        "section_10_exemptions":   float(section_10_exemptions),
        "standard_deduction":      float(standard_deduction),
        "gross_total_income":      float(gross_total_income),
        "total_taxable_income":    float(total_taxable_income),

        # Tax figures
        "tax_paid":                float(net_tax_payable),
        "net_tax_payable":         float(net_tax_payable),
        "tax_on_total_income":     float(tax_on_total_income),
        "health_edu_cess":         float(health_edu_cess),
        "rebate_87a":              float(rebate_87a),

        # Chapter VI-A deductions
        "total_vi_a_deductions":   float(total_vi_a_deductions),
        "deductions_80c":          float(deductions_80c),
        "deductions_80ccd1b":      float(deductions_80ccd1b),
        "deductions_80d":          float(deductions_80d),
        "deductions_80e":          float(deductions_80e),
        "deductions_80g":          float(deductions_80g),
        "deductions_80tta":        float(deductions_80tta),

        # Identity and metadata
        "pan":                     pan,
        "employee_name":           employee_name,
        "employer_name":           employer_name,
        "assessment_year":         assessment_year,
        "period_from":             period_from,
        "period_to":               period_to,
    }

    print(f"Extraction complete:\n{json.dumps(extracted, indent=2, ensure_ascii=False)}")
    return extracted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(file_path: str) -> str:
    """Extract all text from all pages of a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def _get_float(d: dict, keys: list, default: float) -> float:
    """
    Return the first non-zero numeric value found under any of the given keys.
    Skips None, 0, empty string, and non-numeric values.
    """
    for k in keys:
        v = d.get(k)
        if v is None or v == "" or v == 0 or v == 0.0:
            continue
        try:
            f = float(v)
            if f != 0.0:
                return f
        except (ValueError, TypeError):
            continue
    return default


def _get_str(d: dict, keys: list) -> str:
    """Return the first non-empty string value found under any of the given keys."""
    for k in keys:
        v = d.get(k)
        if v and str(v).strip() not in ("", "UNKNOWN", "unknown", "N/A", "null", "None"):
            return str(v).strip()
    return "UNKNOWN"


def _empty_result(reason: str) -> dict:
    """Return a zeroed-out result dict. The reason tag goes into the pan field."""
    return {
        "taxable_salary":          0.0,
        "gross_salary":            0.0,
        "other_employer_salary":   0.0,
        "section_10_exemptions":   0.0,
        "standard_deduction":      50000.0,
        "gross_total_income":      0.0,
        "total_taxable_income":    0.0,
        "tax_paid":                0.0,
        "net_tax_payable":         0.0,
        "tax_on_total_income":     0.0,
        "health_edu_cess":         0.0,
        "rebate_87a":              0.0,
        "total_vi_a_deductions":   0.0,
        "deductions_80c":          0.0,
        "deductions_80ccd1b":      0.0,
        "deductions_80d":          0.0,
        "deductions_80e":          0.0,
        "deductions_80g":          0.0,
        "deductions_80tta":        0.0,
        "pan":                     reason,
        "employee_name":           "UNKNOWN",
        "employer_name":           "UNKNOWN",
        "assessment_year":         "UNKNOWN",
        "period_from":             "UNKNOWN",
        "period_to":               "UNKNOWN",
    }
