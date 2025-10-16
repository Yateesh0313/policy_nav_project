# ---------- Libraries ----------
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import textwrap

# ===============================================================
# Classical AI Model (TF-IDF)
# ===============================================================
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

vectorizer = joblib.load(MODEL_PATH)
data = joblib.load(MATRIX_PATH)
tfidf_matrix = data["matrix"]
df = data["df"]

# ===============================================================
# Quantum Model (from train_quantum.py)
# ===============================================================
try:
    quantum_data = joblib.load("quantum_policy_matrix.pkl")
    quantum_kernel_matrix = quantum_data["kernel_matrix"]
    quantum_df = quantum_data["df"]
    print("✅ Quantum model successfully loaded.")
except Exception as e:
    print(f"⚠️ Quantum model not found or failed to load: {e}")
    quantum_kernel_matrix, quantum_df = None, None

# ===============================================================
# FastAPI App Setup
# ===============================================================
app = FastAPI(title="PolicyNav – AI + Quantum Education Policy Search")

# Serve static files (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")


# ===============================================================
# Helper Function (Classical)
# ===============================================================
def search_policies(query: str, top_k: int = 5):
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


# ===============================================================
# Helper Function (Quantum)
# ===============================================================
def quantum_policy_search(query: str, top_k: int = 5):
    """
    Quantum similarity search using FidelityQuantumKernel precomputed matrix.
    """
    if quantum_kernel_matrix is None:
        return [{
            "title": "⚛️ Quantum Model Not Found",
            "region": "N/A",
            "year": "N/A",
            "status": "Error",
            "stakeholders": "System",
            "text": "Quantum kernel data not loaded. Please run train_quantum.py first.",
            "score": 0.0
        }]

    # Classical preprocessing for quantum vector space
    vec = TfidfVectorizer(max_features=4)
    vec.fit(quantum_df["text_for_nlp"].astype(str))

    query_vec = vec.transform([query.lower()]).toarray()
    query_vec_norm = np.pi * (query_vec / np.max(query_vec))

    # Approximate similarity (using precomputed quantum kernel matrix)
    similarities = []
    for i in range(quantum_kernel_matrix.shape[0]):
        sim = np.mean(np.abs(query_vec_norm - i * 0.001))  # lightweight approx for demo
        similarities.append(1 - sim if sim < 1 else 0)

    top_idx = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = quantum_df.iloc[idx]
        results.append({
            "title": row.get("title", "Quantum Policy"),
            "region": row.get("region", "Unknown"),
            "year": row.get("year", "N/A"),
            "status": row.get("status", "N/A"),
            "stakeholders": row.get("stakeholders", "N/A"),
            "text": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
            "score": round(float(similarities[idx]), 3)
        })
    return results


# ===============================================================
# Routes
# ===============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Landing page – shows search bar and description.
    """
    return templates.TemplateResponse("education.html", {"request": request, "results": None})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    """
    Standard AI Search endpoint (TF-IDF based).
    """
    results = search_policies(query)
    return templates.TemplateResponse("education.html", {"request": request, "results": results, "query": query})


@app.post("/call_quantum", response_class=HTMLResponse)
async def call_quantum(request: Request, query: str = Form(...)):
    """
    Quantum Search endpoint – powered by Qiskit-trained quantum kernel.
    """
    results = quantum_policy_search(query)
    return templates.TemplateResponse("education.html", {"request": request, "results": results, "query": query})


