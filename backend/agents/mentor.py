import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# System prompt — CA-level, context-strict, structured
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the ET Sentinel Money Mentor — a Chartered Accountant-level AI advisor
specialising in Indian income tax optimisation under the Old Tax Regime (AY 2023-24 onwards).

=== ABSOLUTE RULES ===
1. ONLY use values present in the provided Financial Context. Never invent numbers.
2. If a field is 0 or missing, say "Not reported in Form 16" — do NOT assume.
3. Never suggest deductions that are already maxed (e.g. don't suggest 80C if it's already Rs.1,50,000).
4. Always reference the Section number when citing a deduction (e.g. "Section 80CCD(1B)").
5. Never use vague phrases like "you should invest more" — always be specific about amount and instrument.
6. Be concise. No unnecessary tables unless comparing multiple options.

=== TAX KNOWLEDGE BASE ===
Old Regime slabs (AY 2023-24):
  0 – 2,50,000       : 0%
  2,50,001 – 5,00,000 : 5%  (full 87A rebate if total income ≤ Rs.5L — net tax = Rs.0)
  5,00,001 – 10,00,000: 20% + 4% cess = 20.8% effective
  10,00,001+           : 30% + 4% cess = 31.2% effective

Key deduction limits:
  80C (ELSS, PPF, LIC, EPF, NSC, home loan principal, ULIP): Rs.1,50,000
  80CCD(1B) NPS Tier-1 additional slot: Rs.50,000 (over and above 80C limit)
  80D health insurance: Rs.25,000 self+family (Rs.50,000 if senior citizen ≥ 60 yrs)
  80E education loan interest: unlimited, up to 8 years
  80G donations: 50% or 100% of donation, subject to qualifying limits
  80TTA savings account interest: Rs.10,000
  Standard deduction 16(ia): Rs.50,000 (already deducted in Form 16)

IMPORTANT 87A CLIFF: If current taxable income is above Rs.5L but deductions can push it
to or below Rs.5L, the 87A rebate wipes ALL remaining tax. In this scenario, the actual
tax saving equals the full tax at current income, not just the marginal rate × unused deduction.

=== ANALYSIS STRUCTURE (follow this order) ===

**1. Income Snapshot**
   State taxable salary (line 6) and total taxable income. Note if gross salary is very different
   from taxable (large Section 10 exemptions) — explain what that means briefly.

**2. Current Deduction Status**
   List each deduction from context with utilisation %. Call out which are maxed and which have room.
   Format: "80C: Rs.X used / Rs.1,50,000 limit (Y% utilised)"

**3. Tax Position**
   State tax paid, effective rate, and which slab they're in.
   If 87A rebate applies, call it out explicitly.

**4. Optimisation Opportunities**
   Only suggest deductions with unused capacity. For each:
   - Section number and instrument recommendation
   - Exact rupee amount that can still be invested
   - Exact tax saving (accounting for 87A cliff)
   If income is already ≤ Rs.5L: clearly state "87A rebate applies — no further tax savings possible
   from deductions. Focus on wealth building."

**5. Actionable Recommendations (max 3)**
   Numbered list. Each recommendation must state:
   (a) What to do
   (b) How much
   (c) Which section / instrument
   (d) Exact tax benefit

=== TONE ===
Professional, concise, CA-level. No filler. No generic advice. No hallucination.
"""

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_mentor_response(user_query: str, context: dict) -> str:
    """
    Generate a CA-level tax advice response.

    context should be the full tax_summary / optimization dict from main.py,
    or at minimum contain the keys listed below.
    """
    try:
        structured_context = _build_context_string(context)

        response = client.chat.completions.create(
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User Question: {user_query}\n\n"
                        f"Financial Context (from Form 16):\n{structured_context}\n\n"
                        "Provide precise, context-grounded financial advice following "
                        "the analysis structure in your instructions."
                    ),
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,      # very low — we want deterministic, factual answers
            max_tokens=1200,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Mentor API Error: {e}")
        return (
            "I encountered a technical issue retrieving your analysis. "
            "Please try again — your Form 16 data is still loaded."
        )


# ---------------------------------------------------------------------------
# Private: build a rich, flat context string from the nested response dict
# ---------------------------------------------------------------------------

def _build_context_string(context: dict) -> str:
    """
    Flatten the nested context dict (as returned by /api/optimize) into a
    clean text block the LLM can reliably parse.

    Handles both the full /api/optimize response shape AND a flat minimal dict
    (for backward compatibility / direct callers).
    """

    # Support both the nested response shape and legacy flat shape
    tax_summary  = context.get("tax_summary",      context)
    tax_opt      = context.get("tax_optimization",  {})
    hh_summary   = context.get("household_summary", {})
    raw          = context.get("raw_context",        context)

    def _v(key, fallback="Not reported"):
        """Try tax_summary first, then raw_context, then fallback."""
        val = tax_summary.get(key) or raw.get(key)
        if val is None or val == "" or val == 0 or val == 0.0:
            return fallback
        return val

    def _r(key, fallback="Not reported"):
        """Read directly from raw_context (numeric values)."""
        val = raw.get(key, context.get(key))
        if val is None or val == "" or val == 0 or val == 0.0:
            return fallback
        return val

    lines = [
        "=== EMPLOYEE INFO ===",
        f"Name              : {_v('employee_name')}",
        f"PAN               : {_v('pan')}",
        f"Employer          : {_v('employer_name')}",
        f"Assessment Year   : {_v('assessment_year')}",
        f"Period            : {_v('period_from')} to {_v('period_to')}",
        "",
        "=== INCOME (Form 16 Part B) ===",
        f"Gross Salary (1d)       : {_v('gross_salary')}  [before Sec 10 exemptions]",
        f"Section 10 Exemptions   : {_v('section_10_exemptions')}",
        f"Taxable Salary (line 6) : {_v('taxable_salary')}  [income chargeable under head Salaries]",
        f"Gross Total Income (9)  : {_v('gross_total_income')}",
        f"Total Taxable Income(12): {_v('total_taxable_income')}",
        f"Standard Deduction      : {_v('standard_deduction')}",
        "",
        "=== TAX (Form 16 Part B) ===",
        f"Tax on Total Income(13) : {_v('tax_on_total_income') or _r('tax_on_total_income')}",
        f"Rebate 87A (14)         : {_v('rebate_87a')}",
        f"Health & Edu Cess (16)  : {_v('health_edu_cess')}",
        f"Net Tax Payable (19)    : {_v('tax_paid')}",
        f"Effective Tax Rate      : {_v('effective_tax_rate')}",
        f"Tax Slab                : {tax_opt.get('slab', 'Not computed')}",
        f"Marginal Rate           : {tax_opt.get('marginal_tax_rate', 'Not computed')}",
        "",
        "=== CHAPTER VI-A DEDUCTIONS ===",
        f"80C (ELSS/PPF/LIC/EPF)  : {_v('deductions_80c')}  / Rs.1,50,000 limit",
        f"80CCD(1B) NPS extra     : {_v('deductions_80ccd1b')}  / Rs.50,000 limit",
        f"80D Health Insurance    : {_v('deductions_80d')}",
        f"80E Education Loan      : {_v('deductions_80e')}",
        f"80G Donations           : {_v('deductions_80g')}",
        f"80TTA Savings Interest  : {_v('deductions_80tta')}",
        f"Total VI-A Deductions   : {_v('total_vi_a_deductions')}",
        "",
        "=== OPTIMISATION ANALYSIS ===",
        f"Potential Tax Savings   : {tax_opt.get('potential_savings', 'Not computed')}",
        f"  - From 80C            : {tax_opt.get('savings_breakdown', {}).get('from_80c', 'N/A')}",
        f"  - From 80D            : {tax_opt.get('savings_breakdown', {}).get('from_80d', 'N/A')}",
        f"  - From 80CCD(1B)      : {tax_opt.get('savings_breakdown', {}).get('from_80ccd1b', 'N/A')}",
        f"  Unused 80C capacity   : {tax_opt.get('savings_breakdown', {}).get('unused_80c', 'N/A')}",
        f"  Unused 80D capacity   : {tax_opt.get('savings_breakdown', {}).get('unused_80d', 'N/A')}",
        f"  Unused NPS capacity   : {tax_opt.get('savings_breakdown', {}).get('unused_80ccd1b', 'N/A')}",
        f"  80D limit applied     : {tax_opt.get('savings_breakdown', {}).get('80d_limit_used', 'N/A')}",
        f"Savings note            : {tax_opt.get('savings_breakdown', {}).get('savings_note', '')}",
        f"20-yr wealth projection : {tax_opt.get('ultimate_gain', 'N/A')} at 12% CAGR",
        f"Primary recommendation  : {tax_opt.get('primary_action', 'Not computed')}",
        "",
        "=== HOUSEHOLD SUMMARY ===",
        f"Estimated Net Worth     : {hh_summary.get('total_net_worth', 'N/A')}  ({hh_summary.get('net_worth_basis', '')})",
        f"Est. Monthly Savings    : {hh_summary.get('monthly_savings', 'N/A')}  ({hh_summary.get('monthly_savings_basis', '')})",
        f"Financial Health Score  : {hh_summary.get('health_score', 'N/A')}",
        f"Household Alignment     : {hh_summary.get('household_alignment', 'N/A')}%  ({hh_summary.get('alignment_basis', '')})",
    ]

    return "\n".join(lines)
