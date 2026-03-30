# ET Sentinel — AI Money Mentor

> **Transforming Tax Leakage into Generational Wealth**

ET Sentinel converts a user's uploaded Form 16 PDF into a personalised financial strategy in under 10 seconds. Built for the **ET GenAI Hackathon 2026 — Track 9**.

**Live Prototype:** https://etgen-85qg.vercel.app/

---

## The Problem

India has over 80 million salaried taxpayers. The average employee leaves **Rs.46,593 in legal tax savings unclaimed every year** — not through fraud, but through simple ignorance of Section 80C, 80D, and NPS rules. A Chartered Accountant costs Rs.5,000–15,000 per consultation. ET Sentinel delivers CA-level analysis instantly, for free.

---

## System Architecture

ET Sentinel operates on a **Supervisor-Specialist** model: four agents coordinate to ensure zero hallucinated math and traceable advice.

```
Form 16 PDF
     |
     v
[ Librarian ]  ──  pdfplumber + Groq Llama 3.3 70B
     |              Extracts 24 structured fields from Part A & B
     v
[ Tax Engine ]  ── Pure Python (zero LLM)
     |              Deterministic slab calculation, 87A cliff logic,
     |              annuity wealth projection, health score
     v
[ Mentor ]  ────── Groq Llama 3.3 70B
     |              CA-level natural language advice,
     |              grounded in full 24-field context
     v
[ Dashboard ]  ─── Next.js 14 + Recharts
                    Net worth, savings, health score,
                    20-year wealth trajectory, chat interface
```

### Agent Responsibilities

| Agent | Role | Technology |
|---|---|---|
| **Librarian** | Universal Form 16 PDF parser. Extracts 24 fields with multi-key fallbacks and 4 sanity-check chains | `pdfplumber` + Groq Llama 3.3 70B |
| **Tax Engine** | Deterministic Indian income tax calculations. Section 87A cliff, old regime slabs, annuity FV, health score | Pure Python — no LLM for any arithmetic |
| **Mentor** | CA-level contextual advisor. Receives the full 24-field context and pre-computed analysis | Groq Llama 3.3 70B, temperature 0.1 |
| **Dashboard** | Single-page app. Stat cards, 20-year wealth chart, household alignment ring, chat interface | Next.js 14, Tailwind CSS, Recharts |

---

## Financial Logic

### 1. Tax Shield — identifying leakage

The Tax Engine identifies unused deduction capacity across three sections:

```
Unused_80C     = max(1,50,000 − deductions_80c, 0)
Unused_80D     = max(25,000 − deductions_80d, 0)        # 50,000 for senior citizens
Unused_80CCD1B = max(50,000 − deductions_80ccd1b, 0)   # NPS additional slot
```

### 2. Section 87A cliff — the critical edge case

Most fintech tools get this wrong. The simple formula `savings = unused × marginal_rate` overstates savings when deductions push taxable income below Rs.5,00,000, because the Section 87A rebate wipes all remaining tax at that threshold.

ET Sentinel uses a cliff-aware simulation:

```python
def compute_potential_savings(total_taxable_income, unused_80c, unused_80d, unused_80ccd1b):
    # If already at or below Rs.5L, rebate applies — no further savings possible
    if total_taxable_income <= 500000:
        return 0, 0, 0, 0

    # Apply deductions sequentially; savings stop once income hits the Rs.5L floor
    income = total_taxable_income
    for unused in [unused_80ccd1b, unused_80c, unused_80d]:
        income_after = max(income - unused, 0)
        saving = compute_old_regime_tax(income) - compute_old_regime_tax(income_after)
        income = income_after
```

### 3. Wealth Trajectory — annuity future value

Redirecting identified tax leakage into a 12% CAGR instrument compounds as an annuity (annual contributions), not a lump sum:

```
FV = PMT × [(1 + r)^n − 1] / r

Where:
  PMT = annual tax savings (identified leakage)
  r   = 0.12 (12% CAGR — Nifty 50 long-term average)
  n   = 20 years
```

Example (Dileep Singh, Sample_F16.pdf):
- Taxable income: Rs.5,33,457
- Unused 80CCD(1B): Rs.50,000
- Potential savings: Rs.19,959/year
- 20-year FV at 12% CAGR: **Rs.14.4L**

### 4. Health Score — four-dimensional (out of 10)

| Dimension | Max pts | Measure |
|---|---|---|
| 80C utilisation | 4.0 | deductions_80c / 1,50,000 |
| 80D utilisation | 2.0 | deductions_80d / limit (age-adjusted) |
| NPS 80CCD(1B) | 2.0 | deductions_80ccd1b / 50,000 |
| Effective tax rate | 2.0 | tax_paid / taxable_salary (line 6) |

### 5. Net Worth Heuristic

```
Estimated Net Worth = taxable_salary (line 6) × 10
```

Note: uses **taxable salary** (Form 16 Part B, line 6 — income chargeable under head Salaries), not gross salary (line 1d). Gross salary includes Section 10 exemptions and can overstate economic income by 10–40x for high-CTC employees.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, Recharts |
| Backend | FastAPI, Uvicorn, Python 3.13 |
| PDF Extraction | pdfplumber |
| LLM Inference | Groq (Llama 3.3 70B Versatile) — 500+ tokens/sec |
| Deployment | Vercel (frontend), Railway (backend) |

---

## Resilience & Reliability

ET Sentinel is built with a "Zero Dashboard Failure" philosophy:

- **Graceful degradation** — If the Groq API rate-limits, the Librarian returns a structured empty result; the dashboard stays live with a clear error message rather than crashing.
- **Multi-key field extraction** — Each of the 24 Form 16 fields has 4–6 key aliases to catch LLM naming variations (e.g. `"deductions_80c"`, `"Section 80C"`, `"80C"`, `"deduction_80c"`).
- **4 fallback chains** — `total_vi_a_deductions` derived from sum of individual deductions if missing; `net_tax_payable` derived from `tax + cess − rebate` if missing; `taxable_salary` falls back to `gross_total_income`; `total_taxable_income` derived from `taxable_salary − VI-A deductions`.
- **UUID file naming** — uploaded PDFs are prefixed with `uuid4()` to prevent filename collisions under concurrent load.
- **Sanity guards** — if the LLM returns gross salary as taxable salary (a common error on complex Form 16s), the Librarian detects the anomaly and overrides.
- **Structured empty responses** — all error paths return the same response shape as a successful parse, so the frontend never encounters unexpected nulls.

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Groq API key (free tier at console.groq.com)

### Backend

```bash
# Clone the repository
git clone https://github.com/your-username/et-sentinel.git
cd et-sentinel/backend

# Install dependencies
pip install -r requirements.txt

# Create environment file
echo "GROQ_API_KEY=your_key_here" > .env

# Start the server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Test with:
```bash
curl http://localhost:8000/api/health
# {"status":"ok","service":"ET Sentinel"}
```

### Frontend

```bash
cd ../frontend

# Install dependencies
npm install

# Point to your backend
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

Open `http://localhost:3000`.

---

## API Reference

### `POST /api/optimize`

Upload a Form 16 PDF and receive the full analysis.

**Parameters:**
- `files` — Form 16 PDF (multipart/form-data)
- `is_senior_citizen` — boolean query param (default: false). Set true for employees ≥ 60 years; changes 80D limit from Rs.25,000 to Rs.50,000.

**Response shape:**
```json
{
  "household_summary": {
    "total_net_worth": "Rs.68.5L",
    "net_worth_basis": "estimated — 10x taxable salary (line 6)",
    "monthly_savings": "Rs.15,990",
    "health_score": "6.2/10",
    "health_score_breakdown": { "80c_pts": 4.0, "80d_pts": 0.15, "nps_pts": 0.0, "tax_eff_pts": 2.0 },
    "household_alignment": 94
  },
  "tax_optimization": {
    "potential_savings": "Rs.19,959",
    "slab": "20% slab",
    "marginal_tax_rate": "20.8%",
    "savings_breakdown": { "from_80ccd1b": "Rs.19,959", "from_80c": "Rs.0", "from_80d": "Rs.0" },
    "ultimate_gain": "Rs.14.4L",
    "chart_data": [0, 126000, 318000, 637000, 1440000],
    "chart_years": [0, 5, 10, 15, 20],
    "primary_action": "..."
  },
  "tax_summary": { ... },
  "raw_context": { ... }
}
```

### `POST /api/chat`

Chat with the Mentor AI using the loaded Form 16 context.

**Body:**
```json
{
  "message": "Should I invest in NPS or ELSS?",
  "context": { }
}
```

### `GET /api/health`

Liveness check. Returns `{"status": "ok"}`.

---

## Project Structure

```
et-sentinel/
├── backend/
│   ├── main.py                  # FastAPI app, tax engine, all endpoints
│   ├── agents/
│   │   ├── librarian.py         # Form 16 PDF extraction (pdfplumber + Groq)
│   │   ├── mentor.py            # CA-level AI advisor (Groq)
│   │   ├── household_strategist.py
│   │   └── portfolio_auditor.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Dashboard
│   │   ├── couple/page.tsx      # Couple's Planner
│   │   └── portfolio/page.tsx   # Portfolio X-Ray
│   ├── components/
│   └── package.json
└── README.md
```

---

## Macro Impact

| Metric | Value |
|---|---|
| Avg. identified tax savings per user | Rs.46,593/year |
| Advisor cost replaced | Rs.5,000–15,000/year |
| Time to analysis | < 10 seconds |
| Cognitive automation (10,000 users) | ~40,000 hours/month |
| 20-year wealth created per avg. user | Rs.37L+ at 12% CAGR |

---

## Limitations & Disclaimers

- ET Sentinel analyses the **Old Tax Regime** only. New Regime comparison (AY 2024-25 default) is on the roadmap.
- Results are based solely on data present in the uploaded Form 16. Income from other sources (house property, capital gains, business income) is not considered.
- Net worth and monthly savings figures are **estimates** based on standard heuristics, not actual financial statements.
- All AI outputs are for educational purposes only and are not a substitute for advice from a SEBI-registered financial advisor or Chartered Accountant.

---

## Team

Developed by **Amogh, Tansiqa, Atharv, Shravan** for the ET GenAI Hackathon 2026.

---

*ET Sentinel — because every rupee of tax leakage is a rupee that doesn't compound.*
