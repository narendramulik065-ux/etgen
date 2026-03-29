import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are the ET Sentinel Money Mentor, a Chartered Accountant-level AI.

STRICT RULES:
1. ONLY use values provided in the context.
2. DO NOT assume any values like HRA, NPS, Home Loan, etc.
3. If a value is not present in context, say "Not available in Form 16".
4. Your job is to detect financial gaps and suggest optimization.

ANALYSIS STRUCTURE:

1. SALARY ANALYSIS:
- Use salary from context.

2. DEDUCTION ANALYSIS:
- Section 80C used amount
- Check if full (₹1,50,000) or not

3. GAP DETECTION:
- If 80C < 1,50,000 → unused 80C opportunity
- If 80C == 1,50,000 → suggest NPS (₹50,000 under 80CCD(1B))

4. TAX INSIGHT:
- Use tax_paid from context
- Comment if tax is high or optimized

5. FINAL ADVICE:
- Give 2–3 precise actionable suggestions
- NO generic advice

TONE:
- Professional (like CA)
- Clear and concise
- No unnecessary long tables unless needed

IMPORTANT:
- Never hallucinate numbers
- Never assume deductions
- Always refer to context values
"""

def get_mentor_response(user_query: str, context: dict) -> str:
    try:
        # Safely extract context
        salary = context.get("salary", 0)
        tax_paid = context.get("tax_paid", 0)
        deductions_80c = context.get("deductions_80c", 0)

        structured_context = f"""
        Salary: ₹{salary}
        Tax Paid: ₹{tax_paid}
        Section 80C Used: ₹{deductions_80c}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
                    User Question: {user_query}

                    Financial Context:
                    {structured_context}

                    Analyze and give precise financial advice.
                    """
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"⚠️ Mentor API Error: {e}")
        return "Based on your Form 16, I recommend reviewing unused tax-saving opportunities like Section 80C or NPS to optimize your tax liability."
