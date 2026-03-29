# 🛡️ ET Sentinel: AI Money Mentor
### *Transforming Tax Leakage into Generational Wealth*

[cite_start]**ET Sentinel** is a full-stack AI application that converts a user's uploaded Form 16 PDF into a personalized financial strategy[cite: 646, 680]. [cite_start]Developed for the **ET GenAI Hackathon 2026** (Track 9), the platform utilizes a specialized multi-agent pipeline to turn "dead" tax documents into actionable wealth trajectories[cite: 644, 689].

[cite_start]🚀 **Live Prototype:** [https://etgen-85qg.vercel.app/](https://etgen-85qg.vercel.app/) [cite: 679]

---

## 🧠 System Architecture

[cite_start]ET Sentinel operates on a **Supervisor-Specialist** model, coordinating four specialized agents to ensure data integrity and zero "hallucinated math"[cite: 643, 685].

### [cite_start]1. The Agentic Pipeline [cite: 650]
| Agent | Role | Technology |
| :--- | :--- | :--- |
| **Librarian** | Universal PDF parsing and structured data extraction. | `pdfplumber` + Groq Llama 3.1 8B. |
| **Tax Engine** | Deterministic tax calculations and regime-aware gap analysis. | Pure Python (Zero LLM for math). |
| **Strategist** | Household-level optimization and joint tax-saving recommendations. | Rule-based bracket comparison. |
| **Mentor** | Contextual AI advisor providing traceable, step-by-step advice. | Groq Llama 3.1 8B + Markdown Tables. |

### [cite_start]2. The Tech Stack [cite: 677]
* **Frontend**: Next.js 14, Tailwind CSS, Recharts.
* **Backend**: FastAPI, Uvicorn, Python 3.13.
* **Inference**: Groq (Llama 3.1 8B Instant) at 500+ tokens/sec.
* [cite_start]**Deployment**: Vercel (Frontend) and Railway (Backend)[cite: 647].

---

## 📈 Financial Logic & Impact

[cite_start]The core value proposition of ET Sentinel is the **Wealth Trajectory**, which quantifies the lifetime cost of financial inaction[cite: 706].

### 1. The "Tax Shield" Formula
[cite_start]The system identifies **Tax Leakage** by calculating the gap between a user's current deductions and their maximum legal headroom[cite: 701, 703]:
$$Potential Savings = (Unused_{80C} + Unused_{NPS}) \times Tax_{Rate} \times 1.04$$
*(Includes the mandatory 4% Health and Education Cess)*

### [cite_start]2. Wealth Projection [cite: 705, 707]
[cite_start]By redirecting identified leakage into a 12% CAGR instrument, ET Sentinel projects a 20-year wealth trajectory using the compound interest formula[cite: 707, 718]:
$$A = P(1 + r)^n$$
* [cite_start]**Example (Rahul Sharma)**: ₹87,227 annual leakage grows to **₹88.2L** in 20 years[cite: 703, 717].
* **Example (Dileep Singh)**: Identified **₹19,960** leakage grows to **₹1.9L**.

### [cite_start]3. Macro Impact [cite: 695, 699, 704]
* [cite_start]**Cognitive Automation**: Saves **40,000 hours per month** across 10,000 users[cite: 695].
* [cite_start]**Democratization**: Replaces manual advisor costs of **₹5,000/year** with instant AI intelligence[cite: 699].
* [cite_start]**Revenue Recovery**: Identifies an average of **₹46,593** in verified tax savings per tax-paying user[cite: 703, 704].

---

## 🛡️ Resilience & Reliability
[cite_start]ET Sentinel is built with a "Zero Dashboard Failure" philosophy[cite: 670, 671]:
* [cite_start]**Graceful Degradation**: If an AI rate limit is hit, the Librarian falls back to verified sample data to ensure the UI stays live[cite: 672].
* [cite_start]**Deep Search Key-Mapping**: A recursive `deep_search()` function ensures the frontend finds numeric data even if the LLM shifts its JSON key structure[cite: 668, 672].
* [cite_start]**Regime Awareness**: The system automatically detects **Old vs. New Tax Regimes** to provide accurate Health Scores[cite: 672].

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/et-sentinel.git](https://github.com/your-username/et-sentinel.git)
    ```
2.  **Environment Variables**
    Create a `.env` file in the backend root:
    ```env
    GROQ_API_KEY=your_key_here
    ```
3.  **Run Backend (Railway/Local)**
    ```bash
    pip install -r requirements.txt
    uvicorn main:app --reload
    ```
4.  **Run Frontend (Vercel/Local)**
    ```bash
    npm install
    npm run dev
    ```

---

## ⚖️ Disclaimer
[cite_start]*All AI outputs are for educational purposes only and are not a substitute for advice from a SEBI-registered financial advisor[cite: 686].*

---

**Developed by Amogh ,Tansiqa ,atharv,shravan for the ET GenAI Hackathon 2026.**
