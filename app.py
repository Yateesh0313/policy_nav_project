# ---------- Libraries ----------
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# ---------- Load Model + Data ----------
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

vectorizer = joblib.load(MODEL_PATH)
data = joblib.load(MATRIX_PATH)
tfidf_matrix = data["matrix"]
df = data["df"]

# ---------- FastAPI App Setup ----------
app = FastAPI(title="PolicyNav – AI-Powered Policy Search")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- Helper Function ----------
def search_policies(query: str, top_k: int = 6):
    """
    Given a user query, find top-k similar policies using cosine similarity.
    """
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row.get("title", "Untitled Policy"),
            "region": row.get("region", "Unknown"),
            "year": row.get("year", "N/A"),
            "status": row.get("status", "N/A"),
            "stakeholders": row.get("stakeholders", "N/A"),
            "text": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
            "score": round(float(sims[idx]), 3)
        })
    return results

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("education.html", {"request": request, "results": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = search_policies(query)
    return templates.TemplateResponse("education.html", {"request": request, "results": results, "query": query})

@app.post("/call_quantum", response_class=HTMLResponse)
async def call_quantum(request: Request, query: str = Form(...)):
    """
    Quantum Search placeholder – shows purple theme & simulated scores.
    """
    results = [
        {
            "title": "Quantum Policy Optimization for Education Reform",
            "region": "Global",
            "year": "2024",
            "status": "Experimental",
            "score": 0.94,
            "stakeholders": "IBM Qiskit, Policy Research Labs",
            "text": "Uses quantum kernels to improve policy similarity mapping and optimization."
        },
        {
            "title": "Quantum-Enhanced Social Policy Framework",
            "region": "Europe",
            "year": "2023",
            "status": "Research Phase",
            "score": 0.91,
            "stakeholders": "EU Quantum Initiative, Oxford Policy Centre",
            "text": "Explores quantum-assisted algorithms for evaluating social and education policy overlaps."
        },
        {
            "title": "Quantum Curriculum Impact Model",
            "region": "India",
            "year": "2025",
            "status": "Pilot Program",
            "score": 0.89,
            "stakeholders": "IIT Quantum Lab, Ministry of Education",
            "text": "Applies hybrid quantum algorithms to optimize curriculum outcomes across rural schools."
        },
        {
            "title": "Quantum Education Analytics Platform",
            "region": "North America",
            "year": "2024",
            "status": "Deployed",
            "score": 0.87,
            "stakeholders": "Google Quantum AI, Stanford Education Policy Group",
            "text": "Analyzes educational big data using quantum-enhanced data clustering."
        },
        {
            "title": "Quantum Literacy and STEM Equity Initiative",
            "region": "Africa",
            "year": "2023",
            "status": "Active",
            "score": 0.86,
            "stakeholders": "UNESCO, Quantum Africa Initiative",
            "text": "Focuses on quantum literacy for equitable access to STEM education resources."
        },
        {
            "title": "Quantum Policy Impact Simulation System",
            "region": "Australia",
            "year": "2025",
            "status": "Prototype",
            "score": 0.84,
            "stakeholders": "CSIRO Quantum Lab, Policy Analytics Australia",
            "text": "Simulates policy outcomes using quantum kernel-based optimization models."
        }
    ]

    return templates.TemplateResponse(
        "education.html",
        {"request": request, "results": results, "query": query, "quantum_mode": True}
    )
