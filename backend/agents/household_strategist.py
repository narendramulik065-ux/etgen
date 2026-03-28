def get_household_optimization():
    # Synergy Math: Rajesh (30% bracket) & Priya (20% bracket)
    # Savings: ₹30k (HRA) + ₹14k (80C) + ₹2,800 (Loan)
    return {
        "summary": {"joint_net_worth": "₹28.4L", "joint_monthly_savings": "₹92,400", "total_annual_savings": "₹46,800"},
        "comparison": {
            "partner_a": {"name": "Rajesh", "income": "85,000", "tax_bracket": "30%", "hra_status": "Claimed", "80c_usage": "100%"},
            "partner_b": {"name": "Priya", "income": "65,000", "tax_bracket": "20%", "hra_status": "Not Claimed", "80c_usage": "53%"}
        },
        "strategy": [
            {"id": 1, "text": "Transfer HRA claim to Partner B", "impact": "₹30,000/yr"},
            {"id": 2, "text": "Partner B: Maximize 80C to ₹1.5L", "impact": "₹14,000/yr"},
            {"id": 3, "text": "Joint Home Loan: Optimize principal deductions", "impact": "₹2,800/yr"}
        ]
    }
