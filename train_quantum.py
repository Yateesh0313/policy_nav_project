import pandas as pd
import joblib
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

warnings.filterwarnings("ignore")

# ------------------- Load Data -------------------
DATA_PATH = "education_policies.csv"  # CSV file
df = pd.read_csv(DATA_PATH)

# Limit data to first 30 rows for faster training
df = df.head(30)

text_col = "full_text" if "full_text" in df.columns else df.columns[-1]
texts = df[text_col].astype(str).tolist()

print(f"✅ Loaded dataset with {len(texts)} policy records.")
print("🔍 Vectorizing policy texts...")

# ------------------- TF-IDF -------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=4)
tfidf_matrix = vectorizer.fit_transform(texts).toarray()

# ------------------- Quantum Kernel -------------------
print("⚛️ Building Quantum Feature Map and Kernel...")
feature_map = ZZFeatureMap(feature_dimension=vectorizer.max_features, reps=2, entanglement="linear")

# ✅ For Qiskit v1.0+ (no backend or quantum_instance needed)
try:
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
except TypeError:
    # ✅ For older Qiskit (<1.0) compatibility
    from qiskit import Aer
    from qiskit.utils import QuantumInstance
    backend = Aer.get_backend("statevector_simulator")
    qi = QuantumInstance(backend)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, quantum_instance=qi)

# ------------------- Evaluate Kernel -------------------
print("⚙️ Computing quantum similarity matrix (small dataset, should be quick)...")
quantum_sim_matrix = quantum_kernel.evaluate(x_vec=tfidf_matrix)

# ------------------- Save -------------------
joblib.dump(
    {
        "vectorizer": vectorizer,
        "quantum_kernel": quantum_kernel,
        "matrix": quantum_sim_matrix,
        "df": df,
    },
    "quantum_model.pkl",
)

print("✅ Quantum model trained and saved successfully as 'quantum_model.pkl'")
