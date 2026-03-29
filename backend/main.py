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

@app.post("/api/optimize")
async def optimize(files: UploadFile = File(...)): 
    os.makedirs("data", exist_ok=True)
    temp_path = f"data/{files.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(files.file, buffer)
        
    # 1. Get Synchronized Data
    data = get_tax_data(temp_path)
    salary = float(data.get("salary", 0))
    tax = float(data.get("tax_paid", 0))
    deductions = float(data.get("deductions_80c", 0))

    if salary <= 0:
        return {
            "household_summary": {
                "total_net_worth": "₹0L",
                "health_score": "0/10"
            },
            "tax_optimization": {
                "potential_savings": "₹0"
            }
        }

    # 2. Institutional Calculations

    # --- FIX 1: REAL TAX OPTIMIZATION ---
    unused_80c = max(150000 - deductions, 0)

    if salary <= 500000:
        tax_rate = 0.0
    elif salary <= 1000000:
        tax_rate = 0.2
    else:
        tax_rate = 0.3

    potential_savings = int(unused_80c * tax_rate * 1.04)

    # --- FIX 2: REGIME-AWARE HEALTH SCORE ---
    is_new_regime = deductions == 0 and salary > 0

    if is_new_regime:
        if potential_savings == 0:
            h_score = 10.0
        else:
            h_score = 7.5
    else:
        h_score = round((min(deductions, 150000) / 150000) * 10, 1)

    # --- FIX 3: ALIGNMENT LOGIC ---
    leakage_ratio = tax / salary if salary > 0 else 0
    dynamic_alignment = int(100 - (min(leakage_ratio * 200, 40)))

    # --- FIX 4: CORRECT WEALTH PROJECTION ---
    chart_data = [int(potential_savings * (1.12 ** y)) for y in [0, 5, 10, 15, 20]]

    ultimate_gain = (
        f"₹{chart_data[-1] / 10000000:.1f}Cr"
        if chart_data[-1] > 10000000
        else f"₹{chart_data[-1] / 100000:.1f}L"
    )

    return {
        "household_summary": {
            "total_net_worth": f"₹{salary * 1.85 / 100000:.1f}L",
            "monthly_savings": f"₹{int((salary/12)*0.28):,}",
            "health_score": f"{h_score}/10",
            "household_alignment": dynamic_alignment
        },
        "tax_optimization": {
            "potential_savings": f"₹{potential_savings:,}",
            "ultimate_gain": ultimate_gain,
            "chart_data": chart_data,
            "primary_action": f"Redirect ₹{potential_savings:,} unused deduction into {ultimate_gain} lifetime wealth."
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
