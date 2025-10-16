# ============================================================
# Quantum NLP version of TF-IDF pipeline using Qiskit (Optimized for CSV)
# ============================================================

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import joblib
import time

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "quantum_policy_kernel.pkl"
MATRIX_PATH = "quantum_policy_matrix.pkl"

# ----------------------------
# Load datasets (CSV version)
# ----------------------------
print("📘 Loading datasets...")
train_df = pd.read_csv("train_policies.csv")
full_df = pd.read_csv("education_policies.csv")

# ----------------------------
# Preprocess text
# ----------------------------
def preprocess(df):
    df = df.copy()
    df["text_for_nlp"] = (
        df.iloc[:, 1].astype(str) + ". " +
        df["full_text"].astype(str) + ". Stakeholders: " +
        df["stakeholders"].astype(str)
    ).str.lower()
    return df

train_df = preprocess(train_df)
full_df = preprocess(full_df)

# 🧩 For faster local testing (you can remove this limit later)
train_df = train_df.head(10)
full_df = full_df.head(10)

# ----------------------------
# Classical vectorization (TF-IDF)
# ----------------------------
print("⚙️ Computing TF-IDF vectors (reduced features for quantum simulation)...")
vectorizer = TfidfVectorizer(max_features=4)   # reduce if memory/time issues
X_train_tfidf = vectorizer.fit_transform(train_df["text_for_nlp"]).toarray()
X_full_tfidf = vectorizer.transform(full_df["text_for_nlp"]).toarray()

# Normalize values to [0, π] for quantum encoding
X_train_norm = np.pi * (X_train_tfidf / np.max(X_train_tfidf))
X_full_norm = np.pi * (X_full_tfidf / np.max(X_full_tfidf))

# ----------------------------
# Quantum feature map (ZZFeatureMap)
# ----------------------------
n_features = X_full_norm.shape[1]
print(f"🧩 Number of features for quantum map: {n_features}")
feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement='linear')

# ----------------------------
# Quantum kernel computation
# ----------------------------
print("🧠 Computing Quantum Kernel Similarity Matrix... (this may take 1–3 minutes)")
start_time = time.time()

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
kernel_matrix = quantum_kernel.evaluate(X_full_norm, X_full_norm)

end_time = time.time()
elapsed = round(end_time - start_time, 2)
print(f"✅ Quantum kernel computed in {elapsed} seconds")

# ----------------------------
# Save model and matrix
# ----------------------------
joblib.dump(quantum_kernel, MODEL_PATH)
joblib.dump({"kernel_matrix": kernel_matrix, "df": full_df}, MATRIX_PATH)

print(f"\n✅ Quantum kernel model saved to: {MODEL_PATH}")
print(f"✅ Quantum similarity matrix saved to: {MATRIX_PATH}")
print("📊 Matrix shape:", kernel_matrix.shape)
print("🎯 Quantum education policy representation ready!")
