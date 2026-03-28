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
        
    # 1. Get Synchronized Data [cite: 522, 680]
    data = get_tax_data(temp_path)
    salary = float(data.get("salary", 0))
    tax = float(data.get("tax_paid", 0))
    deductions = float(data.get("deductions_80c", 0))

    if salary <= 0:
        return {"household_summary": {"total_net_worth": "₹0L", "health_score": "0/10"}, "tax_optimization": {"potential_savings": "₹0"}}

    # 2. Institutional Calculations [cite: 175, 419, 678]
    h_score = round((min(deductions, 150000) / 150000) * 10, 1)
    
    # NEW: DYNAMIC HOUSEHOLD ALIGNMENT logic 
    # Penalizes the synergy score if tax leakage is high relative to income
    leakage_ratio = tax / salary if salary > 0 else 0
    dynamic_alignment = int(100 - (min(leakage_ratio * 200, 40))) 
    
    # Wealth Trajectory Array: 12% CAGR Growth of tax leakage [cite: 719]
    chart_data = [int(tax * (1.12 ** y)) for y in [0, 5, 10, 15, 20]]
    
    # Standardize the Ultimate Gain string for reuse [cite: 722]
    ultimate_gain = f"₹{chart_data[-1] / 10000000:.1f}Cr" if chart_data[-1] > 10000000 else f"₹{chart_data[-1] / 100000:.1f}L"
    
    return {
        "household_summary": {
            "total_net_worth": f"₹{salary * 1.85 / 100000:.1f}L", 
            "monthly_savings": f"₹{int((salary/12)*0.28):,}",
            "health_score": f"{h_score}/10",
            # FEATURE 1: HOUSEHOLD ALIGNMENT IS NOW DYNAMIC [cite: 720, 729]
            "household_alignment": dynamic_alignment 
        },
        "tax_optimization": {
            "potential_savings": f"₹{int(tax):,}",
            "ultimate_gain": ultimate_gain,
            "chart_data": chart_data,
            # FEATURE 2: REMOVED "" EMPTY GAP [cite: 717, 718]
            "primary_action": f"Redirect ₹{int(tax):,} leakage into {ultimate_gain} lifetime wealth."
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