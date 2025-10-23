# ---------- train_quantum.py ----------
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pennylane as qml
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "education_policies.csv"
df = pd.read_csv(DATA_PATH)
df = df.head(500)  # safe for testing

text_col = "full_text" if "full_text" in df.columns else df.columns[-1]
texts = df[text_col].astype(str).tolist()

print(f"✅ Loaded {len(texts)} policy records")
print("🔍 Building TF-IDF features...")

# -----------------------------
# Classical Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=64)
tfidf_matrix = vectorizer.fit_transform(texts).toarray()
scaler = StandardScaler()
tfidf_matrix_scaled = scaler.fit_transform(tfidf_matrix)

print(f"✅ TF-IDF feature shape: {tfidf_matrix_scaled.shape}")

# -----------------------------
# Dimensionality Reduction for Quantum Embedding
# -----------------------------
n_qubits = 8
pca = PCA(n_components=n_qubits)
tfidf_matrix_reduced = pca.fit_transform(tfidf_matrix_scaled)
print(f"✅ Reduced TF-IDF to {n_qubits} dims for quantum encoding")

# -----------------------------
# Quantum Circuit
# -----------------------------
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i % len(x)], wires=i)
        qml.RZ(x[(i * 2) % len(x)], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def quantum_embed(vec):
    for l in range(n_layers):
        feature_map(vec + 0.5 * l)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] + \
           [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

# -----------------------------
# Generate Quantum Embeddings (Parallelized)
# -----------------------------
print("⚛ Generating quantum embeddings (parallelized)...")
tfidf_matrix_processed = np.tanh(tfidf_matrix_reduced * np.pi)  # compress all at once

def compute_embedding(vec):
    return quantum_embed(vec)

quantum_embeddings = Parallel(n_jobs=-1)(
    delayed(compute_embedding)(vec) for vec in tfidf_matrix_processed
)
quantum_embeddings = np.array(quantum_embeddings)

print(f"✅ Quantum embeddings shape: {quantum_embeddings.shape}")

# -----------------------------
# Compute Similarity Matrix (optional)
# -----------------------------
quantum_sim_matrix = cosine_similarity(quantum_embeddings)
print("🧮 Computed similarity matrix")

# -----------------------------
# Save Model Components
# -----------------------------
joblib.dump({
    "vectorizer": vectorizer,
    "scaler": scaler,
    "pca": pca,
    "embeddings": quantum_embeddings,
    "df": df,
    "n_qubits": n_qubits,
    "n_layers": n_layers
}, "quantum_model.pkl")

print("\n✅ Quantum model trained and saved as quantum_model.pkl")