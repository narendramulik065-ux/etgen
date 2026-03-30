import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from agents.librarian import get_tax_data
from agents.mentor import get_mentor_response
from agents.household_strategist import get_household_optimization
from agents.portfolio_auditor import get_portfolio_analysis

app = FastAPI(title="ET Sentinel — AI Money Mentor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# INDIAN INCOME TAX HELPERS  (Old Regime, AY 2023-24)
# ===========================================================================
#
# Slabs:
#   0       – 2,50,000 :  0%
#   2,50,001 – 5,00,000 :  5%   (but 87A rebate wipes tax if income <= 5L)
#   5,00,001 – 10,00,000 : 20%
#   10,00,001+            : 30%
#   + 4% Health & Education Cess on all computed tax
# ---------------------------------------------------------------------------

def compute_old_regime_tax(taxable_income: float) -> float:
    """Return total tax payable including 4% cess and 87A rebate."""
    if taxable_income <= 250000:
        tax = 0.0
    elif taxable_income <= 500000:
        tax = (taxable_income - 250000) * 0.05
    elif taxable_income <= 1000000:
        tax = 12500.0 + (taxable_income - 500000) * 0.20
    else:
        tax = 12500.0 + 100000.0 + (taxable_income - 1000000) * 0.30

    # Section 87A rebate: zero tax if taxable income <= 5L
    if taxable_income <= 500000:
        tax = 0.0

    return round(tax * 1.04, 2)


def get_marginal_rate(taxable_income: float) -> float:
    """
    Return the effective marginal rate (slab rate x 1.04 cess).
    Returns 0 for income <= 5L because 87A rebate wipes all tax there.
    """
    if taxable_income <= 500000:
        return 0.0           # 87A rebate — no tax savings from more deductions
    elif taxable_income <= 1000000:
        return 0.20 * 1.04   # 20.8%
    else:
        return 0.30 * 1.04   # 31.2%


def compute_potential_savings(
    total_taxable_income: float,
    unused_80c: float,
    unused_80d: float,
    unused_80ccd1b: float,
) -> tuple[int, int, int, int]:
    """
    Compute actual tax savings from unused deductions, correctly handling
    the Section 87A cliff at ₹5,00,000.

    Returns (total_savings, savings_80c, savings_80d, savings_80ccd1b).

    Logic:
      - If income is already <= 5L, all savings = 0 (87A wipes tax).
      - If income is above 5L, deductions reduce tax at marginal rate,
        BUT only down to 5L. Below 5L the saving is zero (rebate kicks in).
      - So effective savings = tax(current) - tax(current - unused_deductions),
        computed properly using compute_old_regime_tax().
    """
    if total_taxable_income <= 500000:
        return 0, 0, 0, 0

    def _saving_for(unused: float) -> int:
        """Tax saved by one deduction block, respecting the 87A floor."""
        if unused <= 0:
            return 0
        income_after = max(total_taxable_income - unused, 0.0)
        return max(int(compute_old_regime_tax(total_taxable_income) -
                       compute_old_regime_tax(income_after)), 0)

    # We must apply deductions in order to track the running income correctly.
    # Compute marginal saving for each block sequentially.
    income_remaining = total_taxable_income

    def _marginal_saving(unused: float, current_income: float) -> tuple[int, float]:
        """Returns (saving_rupees, income_after_deduction)."""
        if unused <= 0 or current_income <= 500000:
            return 0, current_income
        income_after = max(current_income - unused, 0.0)
        saving = max(int(compute_old_regime_tax(current_income) -
                         compute_old_regime_tax(income_after)), 0)
        return saving, income_after

    s_80ccd1b, income_remaining = _marginal_saving(unused_80ccd1b, income_remaining)
    s_80c,     income_remaining = _marginal_saving(unused_80c,     income_remaining)
    s_80d,     income_remaining = _marginal_saving(unused_80d,     income_remaining)

    total = s_80ccd1b + s_80c + s_80d
    return total, s_80c, s_80d, s_80ccd1b


def slab_label(taxable_income: float) -> str:
    """Human-readable slab description."""
    if taxable_income <= 250000:
        return "Nil slab (0%)"
    elif taxable_income <= 500000:
        return "5% slab (87A rebate applies — effective tax 0%)"
    elif taxable_income <= 1000000:
        return "20% slab"
    else:
        return "30% slab"


# ---------------------------------------------------------------------------
# 80D limit by age
# ---------------------------------------------------------------------------

def get_80d_limit(is_senior_citizen: bool = False) -> int:
    """
    80D health insurance deduction limit.
    Self + family: Rs.25,000 (Rs.50,000 if self is senior citizen >= 60 yrs).
    """
    return 50000 if is_senior_citizen else 25000


# ---------------------------------------------------------------------------
# Health Score (out of 10)
# ---------------------------------------------------------------------------

def compute_health_score(
    deductions_80c: float,
    deductions_80d: float,
    deductions_80ccd1b: float,
    total_taxable_income: float,
    tax_paid: float,
    taxable_salary: float,         # FIX: use taxable_salary not gross_salary
    limit_80d: int = 25000,
) -> tuple[float, dict]:
    """
    Score out of 10 across four dimensions:
      Dim 1 — 80C utilisation     : 0-4 pts  (max Rs.1.5L)
      Dim 2 — 80D utilisation     : 0-2 pts  (max Rs.25K / Rs.50K senior)
      Dim 3 — NPS 80CCD(1B)       : 0-2 pts  (max Rs.50K)
      Dim 4 — Effective tax rate  : 0-2 pts  (vs taxable_salary)
    Returns (score, breakdown_dict).
    """
    breakdown = {}

    # Dim 1 — 80C
    d1 = round(min(deductions_80c / 150000, 1.0) * 4.0, 2)
    breakdown["80c_pts"]  = d1
    breakdown["80c_util"] = f"{min(deductions_80c / 150000 * 100, 100):.0f}%"

    # Dim 2 — 80D
    d2 = round(min(deductions_80d / limit_80d, 1.0) * 2.0, 2)
    breakdown["80d_pts"]  = d2
    breakdown["80d_util"] = f"{min(deductions_80d / limit_80d * 100, 100):.0f}%"

    # Dim 3 — 80CCD(1B) / NPS
    d3 = round(min(deductions_80ccd1b / 50000, 1.0) * 2.0, 2)
    breakdown["nps_pts"]  = d3
    breakdown["nps_util"] = f"{min(deductions_80ccd1b / 50000 * 100, 100):.0f}%"

    # Dim 4 — Effective tax rate vs taxable_salary (not gross)
    # FIX: was using gross_salary which inflates the base for high-CTC employees
    base = taxable_salary if taxable_salary > 0 else total_taxable_income
    if base > 0:
        eff_rate = tax_paid / base
        if eff_rate <= 0.03:
            d4 = 2.0
        elif eff_rate <= 0.10:
            d4 = 1.5
        elif eff_rate <= 0.20:
            d4 = 1.0
        else:
            d4 = 0.0
    else:
        d4 = 0.0
    breakdown["tax_eff_pts"]  = d4
    breakdown["eff_tax_rate"] = f"{(tax_paid / base * 100):.2f}%" if base > 0 else "N/A"

    score = round(min(d1 + d2 + d3 + d4, 10.0), 1)
    return score, breakdown


# ---------------------------------------------------------------------------
# Household alignment score (50-100)
# ---------------------------------------------------------------------------

def compute_alignment(tax_paid: float, taxable_salary: float) -> int:
    """
    Measures financial tax efficiency.
    FIX: now uses taxable_salary (line 6) instead of gross_salary (line 1d).
    This prevents artificially high scores on forms where gross is inflated
    by Section 10 exemptions.

    Range: 50 (very high tax leakage) to 100 (near-zero tax).
    """
    if taxable_salary <= 0:
        return 50
    leakage_ratio = tax_paid / taxable_salary
    return int(max(50, 100 - min(leakage_ratio * 200, 50)))


# ---------------------------------------------------------------------------
# 20-year wealth trajectory (annuity formula)
# ---------------------------------------------------------------------------

def compute_wealth_trajectory(
    annual_saving: float,
    cagr: float = 0.12,
    years: int = 20,
) -> tuple[list, float, str]:
    """
    FV of annuity = PMT x [(1+r)^n - 1] / r
    Returns (chart_data at [0,5,10,15,20 yrs], final_fv, display_string).
    """
    if annual_saving <= 0:
        return [0, 0, 0, 0, 0], 0.0, "Rs.0"

    def annuity_fv(pmt, r, n):
        if n == 0:
            return pmt
        return pmt * (((1 + r) ** n - 1) / r)

    milestones = [0, 5, 10, 15, years]
    chart_data = [int(annuity_fv(annual_saving, cagr, y)) for y in milestones]
    final_fv   = annuity_fv(annual_saving, cagr, years)

    if final_fv >= 10_000_000:
        display = f"Rs.{final_fv / 10_000_000:.1f}Cr"
    else:
        display = f"Rs.{final_fv / 100_000:.1f}L"

    return chart_data, final_fv, display


# ===========================================================================
# API ENDPOINTS
# ===========================================================================

@app.post("/api/optimize")
async def optimize(
    files: UploadFile = File(...),
    is_senior_citizen: bool = Query(
        default=False,
        description="True if employee is >= 60 years old (changes 80D limit to Rs.50,000)"
    ),
):
    """
    Upload a Form 16 PDF and receive full tax optimization analysis.
    Query param: is_senior_citizen=true for 60+ employees.
    """

    # --- 1. Save uploaded file with a UUID prefix to avoid collisions ---
    os.makedirs("data", exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{files.filename}"
    temp_path = os.path.join("data", safe_name)

    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(files.file, buf)
    except Exception as e:
        return {"error": f"File save failed: {str(e)}"}

    # --- 2. Extract tax data via the Librarian agent ---
    data = get_tax_data(temp_path)

    # --- 3. Clean up temp file ---
    try:
        os.remove(temp_path)
    except Exception:
        pass

    # --- 4. Pull all extracted values ---
    taxable_salary        = float(data.get("taxable_salary", 0))
    gross_salary          = float(data.get("gross_salary", 0))
    tax_paid              = float(data.get("tax_paid", 0))
    deductions_80c        = float(data.get("deductions_80c", 0))
    deductions_80d        = float(data.get("deductions_80d", 0))
    deductions_80ccd1b    = float(data.get("deductions_80ccd1b", 0))
    deductions_80e        = float(data.get("deductions_80e", 0))
    deductions_80g        = float(data.get("deductions_80g", 0))
    deductions_80tta      = float(data.get("deductions_80tta", 0))
    standard_deduction    = float(data.get("standard_deduction", 50000))
    gross_total_income    = float(data.get("gross_total_income", 0))
    total_taxable_income  = float(data.get("total_taxable_income", 0))
    total_vi_a_deductions = float(data.get("total_vi_a_deductions", 0))
    rebate_87a            = float(data.get("rebate_87a", 0))
    health_edu_cess       = float(data.get("health_edu_cess", 0))
    tax_on_total_income   = float(data.get("tax_on_total_income", 0))

    # --- 5. Guard: if we couldn't extract salary at all ---
    if taxable_salary <= 0 and gross_salary <= 0:
        return {
            "error": (
                "Could not extract salary data from the uploaded document. "
                "Please ensure you have uploaded a valid Form 16 PDF with Part B."
            ),
            "household_summary": _zero_summary(),
            "tax_optimization":  _zero_optimization(),
            "tax_summary":       {},
            "raw_context":       data,
        }

    # --- 6. Derive total_taxable_income if LLM missed it ---
    if total_taxable_income <= 0:
        total_taxable_income = max(taxable_salary - total_vi_a_deductions, 0.0)

    # Use gross_total_income as taxable_salary fallback only
    if taxable_salary <= 0 and gross_total_income > 0:
        taxable_salary = gross_total_income

    # -----------------------------------------------------------------------
    # TAX OPTIMIZATION
    # -----------------------------------------------------------------------

    limit_80d = get_80d_limit(is_senior_citizen)

    # Unused deduction capacity
    unused_80c     = max(150000 - deductions_80c, 0.0)
    unused_80d     = max(limit_80d - deductions_80d, 0.0)
    unused_80ccd1b = max(50000 - deductions_80ccd1b, 0.0)

    # FIX: use cliff-aware savings computation instead of simple rate × unused
    potential_savings, savings_from_80c, savings_from_80d, savings_from_80ccd1b = (
        compute_potential_savings(
            total_taxable_income=total_taxable_income,
            unused_80c=unused_80c,
            unused_80d=unused_80d,
            unused_80ccd1b=unused_80ccd1b,
        )
    )

    # Marginal rate label (for display only)
    marginal_rate = get_marginal_rate(total_taxable_income)

    # -----------------------------------------------------------------------
    # WEALTH TRAJECTORY (annuity, 20 yr @ 12% CAGR)
    # -----------------------------------------------------------------------
    CAGR  = 0.12
    YEARS = 20
    chart_data, final_fv, ultimate_gain = compute_wealth_trajectory(
        annual_saving=potential_savings,
        cagr=CAGR,
        years=YEARS,
    )

    # Primary action message
    if potential_savings > 0:
        action_parts = []
        if savings_from_80ccd1b > 0:
            action_parts.append(f"Rs.{savings_from_80ccd1b:,} via NPS 80CCD(1B)")
        if savings_from_80c > 0:
            action_parts.append(f"Rs.{savings_from_80c:,} via 80C (ELSS/PPF/LIC)")
        if savings_from_80d > 0:
            action_parts.append(f"Rs.{savings_from_80d:,} via 80D (Health Insurance)")
        detail = " + ".join(action_parts) if action_parts else f"Rs.{potential_savings:,}"
        primary_action = (
            f"Save {detail} in taxes this year "
            f"— grows to {ultimate_gain} over {YEARS} years at {int(CAGR * 100)}% CAGR."
        )
    else:
        primary_action = (
            "Your income is at or below the Rs.5L threshold — Section 87A rebate "
            "eliminates all tax. Additional deductions won't reduce your tax further. "
            "Consider a CA for wealth-building strategies like ELSS SIPs or NPS for retirement."
        )

    # -----------------------------------------------------------------------
    # HEALTH SCORE
    # -----------------------------------------------------------------------
    h_score, score_breakdown = compute_health_score(
        deductions_80c=deductions_80c,
        deductions_80d=deductions_80d,
        deductions_80ccd1b=deductions_80ccd1b,
        total_taxable_income=total_taxable_income,
        tax_paid=tax_paid,
        taxable_salary=taxable_salary,     # FIX: pass taxable_salary, not gross
        limit_80d=limit_80d,
    )

    # -----------------------------------------------------------------------
    # HOUSEHOLD ALIGNMENT
    # FIX: use taxable_salary as base, not gross_salary
    # -----------------------------------------------------------------------
    dynamic_alignment = compute_alignment(tax_paid, taxable_salary)

    # -----------------------------------------------------------------------
    # NET WORTH — heuristic: 10x taxable_salary
    # FIX: was using gross_salary (line 1d) which includes Section 10 exemptions
    # and wildly overstates take-home pay. taxable_salary (line 6) is the correct
    # proxy for annual economic income.
    # -----------------------------------------------------------------------
    nw_base = taxable_salary  # line 6: income chargeable under head Salaries
    net_worth_value   = nw_base * 10
    net_worth_display = (
        f"Rs.{net_worth_value / 10_000_000:.1f}Cr"
        if net_worth_value >= 10_000_000
        else f"Rs.{net_worth_value / 100_000:.1f}L"
    )

    # -----------------------------------------------------------------------
    # MONTHLY SAVINGS — 28% savings rate on taxable salary
    # -----------------------------------------------------------------------
    monthly_savings_amount = int((taxable_salary / 12) * 0.28)

    # -----------------------------------------------------------------------
    # CROSS-CHECK: computed tax vs Form 16 tax
    # -----------------------------------------------------------------------
    computed_tax = compute_old_regime_tax(total_taxable_income)
    tax_diff     = round(abs(computed_tax - tax_paid), 2)

    # -----------------------------------------------------------------------
    # BUILD RESPONSE
    # -----------------------------------------------------------------------
    return {
        "household_summary": {
            "total_net_worth":        net_worth_display,
            "net_worth_basis":        "estimated — 10x taxable salary (line 6, income chargeable under head Salaries)",
            "monthly_savings":        f"Rs.{monthly_savings_amount:,}",
            "monthly_savings_basis":  "estimated — assumed 28% savings rate on taxable salary",
            "health_score":           f"{h_score}/10",
            "health_score_breakdown": score_breakdown,
            "household_alignment":    dynamic_alignment,
            "alignment_basis":        "based on effective tax rate vs taxable salary (line 6)",
        },

        "tax_optimization": {
            "potential_savings":      f"Rs.{potential_savings:,}",
            "marginal_tax_rate":      f"{marginal_rate * 100:.1f}%",
            "slab":                   slab_label(total_taxable_income),
            "savings_breakdown": {
                "from_80c":           f"Rs.{savings_from_80c:,}",
                "from_80d":           f"Rs.{savings_from_80d:,}",
                "from_80ccd1b":       f"Rs.{savings_from_80ccd1b:,}",
                "unused_80c":         f"Rs.{unused_80c:,.0f}",
                "unused_80d":         f"Rs.{unused_80d:,.0f}",
                "unused_80ccd1b":     f"Rs.{unused_80ccd1b:,.0f}",
                "80d_limit_used":     f"Rs.{limit_80d:,} ({'senior citizen' if is_senior_citizen else 'standard'})",
                "savings_note": (
                    "Savings correctly account for Section 87A rebate cliff at Rs.5,00,000. "
                    "Deductions that push taxable income below Rs.5L yield zero additional savings."
                ),
            },
            "ultimate_gain":          ultimate_gain,
            "ultimate_gain_basis":    (
                f"Annual Rs.{potential_savings:,} reinvested at {int(CAGR * 100)}% CAGR "
                f"for {YEARS} years — annuity future value formula"
            ),
            "chart_data":             chart_data,
            "chart_years":            [0, 5, 10, 15, 20],
            "primary_action":         primary_action,
        },

        "tax_summary": {
            # Identity
            "employee_name":          data.get("employee_name", ""),
            "employer_name":          data.get("employer_name", ""),
            "pan":                    data.get("pan", ""),
            "assessment_year":        data.get("assessment_year", ""),
            "period_from":            data.get("period_from", ""),
            "period_to":              data.get("period_to", ""),

            # Income
            "gross_salary":           f"Rs.{gross_salary:,.0f}",
            "gross_salary_note":      "Line 1(d) — total before Section 10 exemptions",
            "section_10_exemptions":  f"Rs.{float(data.get('section_10_exemptions', 0)):,.0f}",
            "taxable_salary":         f"Rs.{taxable_salary:,.0f}",
            "taxable_salary_note":    "Line 6 — income chargeable under head Salaries",
            "gross_total_income":     f"Rs.{gross_total_income:,.0f}",
            "total_taxable_income":   f"Rs.{total_taxable_income:,.0f}",
            "standard_deduction":     f"Rs.{standard_deduction:,.0f}",

            # Tax
            "tax_paid":               f"Rs.{tax_paid:,.0f}",
            "computed_tax":           f"Rs.{computed_tax:,.2f}",
            "tax_variance":           f"Rs.{tax_diff:,.2f}",
            "rebate_87a":             f"Rs.{rebate_87a:,.0f}",
            "health_edu_cess":        f"Rs.{health_edu_cess:,.0f}",
            "effective_tax_rate":     (
                f"{(tax_paid / taxable_salary * 100):.2f}%"
                if taxable_salary > 0 else "N/A"
            ),
            "effective_tax_rate_note": "tax paid / taxable salary (line 6)",

            # Deductions
            "total_vi_a_deductions":  f"Rs.{total_vi_a_deductions:,.0f}",
            "deductions_80c":         f"Rs.{deductions_80c:,.0f}",
            "deductions_80ccd1b":     f"Rs.{deductions_80ccd1b:,.0f}",
            "deductions_80d":         f"Rs.{deductions_80d:,.0f}",
            "deductions_80e":         f"Rs.{deductions_80e:,.0f}",
            "deductions_80g":         f"Rs.{deductions_80g:,.0f}",
            "deductions_80tta":       f"Rs.{deductions_80tta:,.0f}",
        },

        "raw_context": data,
    }


@app.post("/api/chat")
async def chat(request: Request):
    """Mentor AI chat — send a user message with the tax context."""
    body    = await request.json()
    message = body.get("message", "")
    context = body.get("context", {})
    if not message:
        return {"reply": "Please send a message."}
    return {"reply": get_mentor_response(message, context)}


@app.post("/api/couple/analyze")
async def couple_analyze():
    """Couple / household financial planner analysis."""
    return get_household_optimization()


@app.post("/api/portfolio/analyze")
async def portfolio_analyze():
    """Portfolio X-Ray analysis."""
    return get_portfolio_analysis()


@app.get("/api/health")
async def health_check():
    """Simple liveness check."""
    return {"status": "ok", "service": "ET Sentinel"}


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _zero_summary() -> dict:
    return {
        "total_net_worth":        "Rs.0",
        "net_worth_basis":        "insufficient data",
        "monthly_savings":        "Rs.0",
        "monthly_savings_basis":  "insufficient data",
        "health_score":           "0/10",
        "health_score_breakdown": {},
        "household_alignment":    0,
        "alignment_basis":        "insufficient data",
    }


def _zero_optimization() -> dict:
    return {
        "potential_savings":   "Rs.0",
        "marginal_tax_rate":   "0%",
        "slab":                "unknown",
        "savings_breakdown":   {},
        "ultimate_gain":       "Rs.0",
        "ultimate_gain_basis": "",
        "chart_data":          [0, 0, 0, 0, 0],
        "chart_years":         [0, 5, 10, 15, 20],
        "primary_action":      "Unable to process document. Please upload a valid Form 16 PDF.",
    }
