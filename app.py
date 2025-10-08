# ================================
# 📘 PolicyNav - Education Policy Search (FastAPI)
# ================================

# ---------- Imports ----------
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
app = FastAPI()

# Serve static files (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates directory
templates = Jinja2Templates(directory="templates")


# ---------- Core Search Function ----------
def search_policies(query: str, top_k: int = 3):
    """Return top_k most relevant policies based on query."""
    # Transform the query into TF-IDF vector
    query_vec = vectorizer.transform([query.lower()])

    # Compute cosine similarity
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of top-k similar policies
    top_idx = sims.argsort()[::-1][:top_k]

    # Format the top results
    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row["title"],
            "policy_id": row["policy_id"],
            "region": row["region"],
            "year": row["year"],
            "status": row["status"],
            "summary": textwrap.shorten(str(row["full_text"]), width=250, placeholder="..."),
            "score": round(float(sims[idx]), 3)
        })
    return results


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render homepage (Education Policy Search)."""
    return templates.TemplateResponse(
        "education.html",  # Use your template file name here
        {"request": request, "results": None}
    )


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    """Handle user query and display search results."""
    results = search_policies(query)
    return templates.TemplateResponse(
        "education.html",  # Use your template file name here
        {"request": request, "results": results, "query": query}
    )


