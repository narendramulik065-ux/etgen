import os
import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from agents.librarian import get_tax_data
from agents.mentor import get_mentor_response
from agents.household_strategist import get_household_optimization
from agents.portfolio_auditor import get_portfolio_analysis

app = FastAPI(title="ET Sentinel Final")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Indian Income Tax Slabs — Old Regime (AY 2023-24 onwards)
# Applied on TAXABLE income (after all deductions)
# ---------------------------------------------------------------------------

def compute_old_regime_tax(taxable_income: float) -> float:
    """
    Compute income tax under the Old Tax Regime.
    Slabs (AY 2023-24):
      0      – 2,50,000 : 0%
      2,50,001 – 5,00,000 : 5%
      5,00,001 – 10,00,000 : 20%
      10,00,001+           : 30%
    + 4% Health & Education Cess on the computed tax.
    Rebate u/s 87A: if taxable income <= 5,00,000, full tax rebate (tax = 0).
    """
    tax = 0.0

    if taxable_income <= 250000:
        tax = 0.0
    elif taxable_income <= 500000:
        tax = (taxable_income - 250000) * 0.05
    elif taxable_income <= 1000000:
        tax = 12500 + (taxable_income - 500000) * 0.20
    else:
        tax = 12500 + 100000 + (taxable_income - 1000000) * 0.30

    # Rebate u/s 87A — if taxable income <= 5L, tax becomes 0
    if taxable_income <= 500000:
        tax = 0.0

    # 4% Health & Education Cess
    tax = tax * 1.04

    return round(tax, 2)


def get_marginal_rate(taxable_income: float) -> float:
    """
    Return the marginal tax rate (including cess) for the given taxable income.
    Used to calculate the value of one additional rupee of deduction.
    """
    if taxable_income <= 250000:
        return 0.0
    elif taxable_income <= 500000:
        return 0.05 * 1.04   # 5.2%
    elif taxable_income <= 1000000:
        return 0.20 * 1.04   # 20.8%
    else:
        return 0.30 * 1.04   # 31.2%


# ---------------------------------------------------------------------------
# Health Score — based on utilization of available deductions
# ---------------------------------------------------------------------------

def compute_health_score(
    deductions_80c: float,
    deductions_80d: float,
    deductions_80ccd1b: float,
    taxable_income: float,
    tax_paid: float,
    gross_salary: float,
) -> float:
    """
    Score out of 10 based on:
      - 80C utilization (max ₹1.5L)     → 4 points
      - 80D utilization (max ₹25K)       → 2 points
      - 80CCD(1B) NPS usage (max ₹50K)   → 2 points
      - Effective tax rate reasonableness → 2 points
    """
    score = 0.0

    # 80C: up to 4 points
    score += min(deductions_80c / 150000, 1.0) * 4.0

    # 80D: up to 2 points
    score += min(deductions_80d / 25000, 1.0) * 2.0

    # 80CCD(1B): up to 2 points
    score += min(deductions_80ccd1b / 50000, 1.0) * 2.0

    # Effective tax rate: 2 points if tax rate looks reasonable
    # (not over-paying due to missing deductions)
    if gross_salary > 0:
        effective_rate = tax_paid / gross_salary
        if effective_rate <= 0.05:
            score += 2.0
        elif effective_rate <= 0.15:
            score += 1.0
        else:
            score += 0.0

    return round(min(score, 10.0), 1)


# ---------------------------------------------------------------------------
# Main API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/optimize")
async def optimize(files: UploadFile = File(...)):
    # --- Save uploaded file ---
    os.makedirs("data", exist_ok=True)
    temp_path = f"data/{files.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(files.file, buffer)
    except Exception as e:
        return {"error": f"File save failed: {str(e)}"}

    # --- Extract tax data via librarian ---
    data = get_tax_data(temp_path)

    # --- Clean up temp file ---
    try:
        os.remove(temp_path)
    except Exception:
        pass

    # Use TAXABLE salary (after sec 10 exemptions) for all tax math
    taxable_salary   = float(data.get("taxable_salary", 0))
    gross_salary     = float(data.get("gross_salary", 0))
    tax_paid         = float(data.get("tax_paid", 0))
    deductions_80c   = float(data.get("deductions_80c", 0))
    deductions_80d   = float(data.get("deductions_80d", 0))
    deductions_80ccd1b = float(data.get("deductions_80ccd1b", 0))
    standard_deduction = float(data.get("standard_deduction", 50000))
    total_taxable_income = float(data.get("total_taxable_income", 0))
    total_vi_a_deductions = float(data.get("total_vi_a_deductions", 0))

    # Fallback: if taxable_salary is 0, we can't do anything meaningful
    if taxable_salary <= 0 and gross_salary <= 0:
        return {
            "error": "Could not extract salary data from the document.",
            "household_summary": {
                "total_net_worth": "₹0L",
                "monthly_savings": "₹0",
                "health_score": "0/10",
                "household_alignment": 0
            },
            "tax_optimization": {
                "potential_savings": "₹0",
                "ultimate_gain": "₹0",
                "chart_data": [0, 0, 0, 0, 0],
                "primary_action": "Unable to process document. Please upload a valid Form 16."
            },
            "raw_context": data
        }

    # --- Derive total taxable income if not extracted directly ---
    # total_taxable_income = taxable_salary - VI-A deductions
    if total_taxable_income <= 0:
        total_taxable_income = max(taxable_salary - total_vi_a_deductions, 0)

    # -------------------------------------------------------------------
    # TAX OPTIMIZATION — identify unused deductions
    # -------------------------------------------------------------------

    # Unused 80C (max ₹1,50,000)
    unused_80c = max(150000 - deductions_80c, 0)

    # Unused 80D — assume standard limit ₹25,000 (self); ₹50,000 for senior citizen
    # We default to ₹25,000 limit here
    unused_80d = max(25000 - deductions_80d, 0)

    # Unused 80CCD(1B) — NPS additional ₹50,000 over and above 80C limit
    unused_80ccd1b = max(50000 - deductions_80ccd1b, 0)

    # Total unused deductions
    total_unused = unused_80c + unused_80d + unused_80ccd1b

    # Marginal rate based on TAXABLE income (correct slab calculation)
    marginal_rate = get_marginal_rate(total_taxable_income)

    # Potential tax savings = unused deductions × marginal rate
    potential_savings = int(total_unused * marginal_rate)

    # Breakdown for transparency
    savings_from_80c      = int(unused_80c * marginal_rate)
    savings_from_80d      = int(unused_80d * marginal_rate)
    savings_from_80ccd1b  = int(unused_80ccd1b * marginal_rate)

    # -------------------------------------------------------------------
    # HEALTH SCORE
    # -------------------------------------------------------------------
    h_score = compute_health_score(
        deductions_80c=deductions_80c,
        deductions_80d=deductions_80d,
        deductions_80ccd1b=deductions_80ccd1b,
        taxable_income=total_taxable_income,
        tax_paid=tax_paid,
        gross_salary=gross_salary if gross_salary > 0 else taxable_salary,
    )

    # -------------------------------------------------------------------
    # HOUSEHOLD ALIGNMENT
    # Measures tax efficiency: lower effective tax rate = better alignment
    # Penalty capped at 50 points so range is 50–100
    # -------------------------------------------------------------------
    base_salary_for_alignment = gross_salary if gross_salary > 0 else taxable_salary
    if base_salary_for_alignment > 0:
        leakage_ratio = tax_paid / base_salary_for_alignment
        # Scale: 0% effective tax → 100%, 25%+ effective tax → 50%
        dynamic_alignment = int(100 - min(leakage_ratio * 200, 50))
    else:
        dynamic_alignment = 50

    # -------------------------------------------------------------------
    # WEALTH TRAJECTORY (20-year projection)
    # Uses annuity formula: FV = PMT × [(1+r)^n - 1] / r
    # Assumes the tax saving is reinvested every year for 20 years @ 12% CAGR
    # -------------------------------------------------------------------
    CAGR = 0.12
    YEARS = 20

    if potential_savings > 0:
        # Annuity FV — annual saving reinvested for N years
        annuity_fv = potential_savings * ((((1 + CAGR) ** YEARS) - 1) / CAGR)

        # Chart shows cumulative value at each milestone year using annuity formula
        chart_data = [
            int(potential_savings * ((((1 + CAGR) ** y) - 1) / CAGR)) if y > 0 else potential_savings
            for y in [0, 5, 10, 15, 20]
        ]

        ultimate_gain = (
            f"₹{annuity_fv / 10000000:.1f}Cr"
            if annuity_fv > 10000000
            else f"₹{annuity_fv / 100000:.1f}L"
        )

        # Build primary action message with savings breakdown
        action_parts = []
        if savings_from_80c > 0:
            action_parts.append(f"₹{savings_from_80c:,} via 80C (ELSS/PPF/LIC)")
        if savings_from_80ccd1b > 0:
            action_parts.append(f"₹{savings_from_80ccd1b:,} via NPS 80CCD(1B)")
        if savings_from_80d > 0:
            action_parts.append(f"₹{savings_from_80d:,} via 80D (Health Insurance)")

        action_detail = " + ".join(action_parts) if action_parts else f"₹{potential_savings:,}"
        primary_action = (
            f"Save {action_detail} in taxes this year → "
            f"grows to {ultimate_gain} over 20 years at {int(CAGR*100)}% CAGR."
        )
    else:
        chart_data = [0, 0, 0, 0, 0]
        ultimate_gain = "₹0"
        primary_action = (
            "All major deductions (80C, 80D, NPS) are already fully utilized. "
            "Consider consulting a CA for advanced tax planning strategies."
        )

    # -------------------------------------------------------------------
    # NET WORTH — only show if we have gross salary
    # We use a conservative 10x annual income heuristic, clearly labeled
    # -------------------------------------------------------------------
    if gross_salary > 0:
        # Approximate net worth: 10x gross annual (industry heuristic for mid-career)
        # Label this clearly as "estimated" in the UI
        net_worth_display = f"₹{gross_salary * 10 / 100000:.1f}L"
        net_worth_basis = "estimated (10x gross salary heuristic)"
    elif taxable_salary > 0:
        net_worth_display = f"₹{taxable_salary * 10 / 100000:.1f}L"
        net_worth_basis = "estimated (10x taxable salary heuristic)"
    else:
        net_worth_display = "N/A"
        net_worth_basis = "insufficient data"

    # Monthly savings: 28% of (taxable_salary / 12) — labeled as assumed
    monthly_savings_amount = int((taxable_salary / 12) * 0.28) if taxable_salary > 0 else 0

    return {
        "household_summary": {
            "total_net_worth": net_worth_display,
            "net_worth_basis": net_worth_basis,
            "monthly_savings": f"₹{monthly_savings_amount:,}",
            "monthly_savings_basis": "assumed 28% savings rate",
            "health_score": f"{h_score}/10",
            "household_alignment": dynamic_alignment,
        },
        "tax_optimization": {
            "potential_savings": f"₹{potential_savings:,}",
            "savings_breakdown": {
                "from_80c":      f"₹{savings_from_80c:,}",
                "from_80d":      f"₹{savings_from_80d:,}",
                "from_80ccd1b":  f"₹{savings_from_80ccd1b:,}",
                "unused_80c":    f"₹{unused_80c:,}",
                "unused_80d":    f"₹{unused_80d:,}",
                "unused_80ccd1b":f"₹{unused_80ccd1b:,}",
            },
            "marginal_tax_rate": f"{marginal_rate * 100:.1f}%",
            "ultimate_gain": ultimate_gain,
            "ultimate_gain_basis": f"Annual ₹{potential_savings:,} reinvested at {int(CAGR*100)}% CAGR for {YEARS} years (annuity)",
            "chart_data": chart_data,
            "primary_action": primary_action,
        },
        "tax_summary": {
            "taxable_salary":        f"₹{taxable_salary:,.0f}",
            "gross_salary":          f"₹{gross_salary:,.0f}",
            "total_taxable_income":  f"₹{total_taxable_income:,.0f}",
            "tax_paid":              f"₹{tax_paid:,.0f}",
            "effective_tax_rate":    f"{(tax_paid/gross_salary*100):.2f}%" if gross_salary > 0 else "N/A",
            "deductions_80c":        f"₹{deductions_80c:,.0f}",
            "deductions_80d":        f"₹{deductions_80d:,.0f}",
            "deductions_80ccd1b":    f"₹{deductions_80ccd1b:,.0f}",
            "employee_name":         data.get("employee_name", ""),
            "pan":                   data.get("pan", ""),
            "assessment_year":       data.get("assessment_year", ""),
        },
        "raw_context": data
    }


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    return {"reply": get_mentor_response(body["message"], body["context"])}


@app.post("/api/couple/analyze")
async def couple_analyze():
    return get_household_optimization()


@app.post("/api/portfolio/analyze")
async def portfolio_analyze():
    return get_portfolio_analysis()
