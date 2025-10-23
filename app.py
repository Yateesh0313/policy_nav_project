# ---------- app.py ----------
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pennylane as qml

# -----------------------------
# App Setup
# -----------------------------
app = FastAPI(title="PolicyNav – AI-Powered Policy Search")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Classical Model Loading
# -----------------------------
df = pd.read_csv("education_policies.csv")
df["combined_text"] = (
    df["title"].astype(str) + " " +
    df["summary"].astype(str) + " " +
    df["goals"].astype(str) + " " +
    df["full_text"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

# -----------------------------
# Quantum Model Loading
# -----------------------------
quantum_data = joblib.load("quantum_model.pkl")
quantum_embeddings = quantum_data["embeddings"]
quantum_vectorizer = quantum_data["vectorizer"]
quantum_scaler = quantum_data["scaler"]
quantum_pca = quantum_data["pca"]
quantum_df = quantum_data["df"]
n_qubits = quantum_data["n_qubits"]
n_layers = quantum_data["n_layers"]

dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i % len(x)], wires=i)
        qml.RZ(x[(i * 2) % len(x)], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def quantum_embed(x):
    for l in range(n_layers):
        feature_map(x + 0.5 * l)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("education.html", {"request": request, "results": None})

# ---------- Classical Search ----------
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df["score"] = sims
    top_results = df.sort_values("score", ascending=False).head(6)

    results = []
    for _, row in top_results.iterrows():
        results.append({
            "title": row["title"],
            "region": row["region"],
            "year": row["year"],
            "status": row["status"],
            "stakeholders": row["stakeholders"],
            "score": round(float(row["score"]), 3),
            "text": row["summary"]
        })

    return templates.TemplateResponse("education.html", {
        "request": request,
        "results": results,
        "query": query,
        "quantum_mode": False
    })

# ---------- Quantum Search ----------
@app.post("/call_quantum", response_class=HTMLResponse)
async def call_quantum(request: Request, query: str = Form(...)):
    try:
        # 1️⃣ Transform query with trained vectorizer
        query_vec = quantum_vectorizer.transform([query]).toarray()
        query_scaled = quantum_scaler.transform(query_vec)
        query_reduced = quantum_pca.transform(query_scaled)

        # 2️⃣ Truncate or match qubits
        query_input = query_reduced[0][:n_qubits]

        # 3️⃣ Quantum embedding
        query_emb = np.array(quantum_embed(query_input)).reshape(1, -1)

        # 4️⃣ Cosine similarity
        sims = cosine_similarity(query_emb, quantum_embeddings).flatten()
        quantum_df["score"] = sims
        top_results = quantum_df.sort_values("score", ascending=False).head(6)

        results = []
        for _, row in top_results.iterrows():
            results.append({
                "title": row["title"],
                "region": row["region"],
                "year": row["year"],
                "status": row["status"],
                "stakeholders": row["stakeholders"],
                "score": round(float(row["score"]), 3),
                "text": row.get("summary", "")
            })

        return templates.TemplateResponse("education.html", {
            "request": request,
            "results": results,
            "query": query,
            "quantum_mode": True
        })

    except Exception as e:
        print("❌ Quantum search error:", e)
        return templates.TemplateResponse("education.html", {
            "request": request,
            "results": None,
            "query": query,
            "quantum_mode": True,
            "error": str(e)
        })