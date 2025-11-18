import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import pennylane as qml
import mysql.connector

# ---------------------------------------------------
# App Setup
# ---------------------------------------------------
app = FastAPI(title="PolicyNav – AI-Powered Policy Search")

app.add_middleware(
    SessionMiddleware,
    secret_key="secret123",
    max_age=None
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Yateesh@12",   # <-- CHANGE THIS
        database="policynav"
    )


# ---------------------------------------------------
# Classical Model Loading
# ---------------------------------------------------
df = pd.read_csv("education_policies.csv")

df["combined_text"] = (
    df["title"].astype(str) + " " +
    df["summary"].astype(str) + " " +
    df["goals"].astype(str) + " " +
    df["full_text"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])


# ---------------------------------------------------
# Quantum Model Loading
# ---------------------------------------------------
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


# ---------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


# ---------------------------------------------------
# LOGIN AUTH USING MYSQL
# ---------------------------------------------------
@app.post("/login_auth", response_class=HTMLResponse)
async def login_auth(request: Request, username: str = Form(...), password: str = Form(...)):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()

    if user:
        request.session["user"] = username
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Invalid username or password"
    })


# ---------------------------------------------------
# REGISTER USING MYSQL
# ---------------------------------------------------
@app.post("/register_auth", response_class=HTMLResponse)
async def register_auth(request: Request, username: str = Form(...), password: str = Form(...)):

    db = get_db()
    cursor = db.cursor()

    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        db.commit()

    except mysql.connector.Error:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "reg_error": "Username already exists"
        })

    return templates.TemplateResponse("login.html", {
        "request": request,
        "success": "Account created! Please login."
    })


# ---------------------------------------------------
# LOGOUT
# ---------------------------------------------------
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ---------------------------------------------------
# HOME (PROTECTED)
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse("education.html", {
        "request": request,
        "results": None
    })


# ---------------------------------------------------
# Classical Search
# ---------------------------------------------------
@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):

    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

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
            "text": row["summary"],
        })

    return templates.TemplateResponse("education.html", {
        "request": request,
        "results": results,
        "query": query,
        "quantum_mode": False
    })


# ---------------------------------------------------
# Quantum Search
# ---------------------------------------------------
@app.post("/call_quantum", response_class=HTMLResponse)
async def call_quantum(request: Request, query: str = Form(...)):

    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    try:
        query_vec = quantum_vectorizer.transform([query]).toarray()
        query_scaled = quantum_scaler.transform(query_vec)
        query_reduced = quantum_pca.transform(query_scaled)
        query_input = query_reduced[0][:n_qubits]

        query_emb = np.array(quantum_embed(query_input)).reshape(1, -1)

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
        return templates.TemplateResponse("education.html", {
            "request": request,
            "results": None,
            "query": query,
            "quantum_mode": True,
            "error": str(e)
        })
