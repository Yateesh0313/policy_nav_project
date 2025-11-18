
# PolicyNav – AI-Powered Policy Search (Education + Poverty)

PolicyNav is an AI-powered web application that helps users explore and search government policies related to **Education** and **Poverty** using:

✔ Classical Machine Learning (TF-IDF + Cosine Similarity)  
✔ Quantum Machine Learning (Pennylane Quantum Embeddings)  
✔ FastAPI Backend  
✔ Secure Login + Register with CSV user storage  
✔ Interactive UI with Visual Charts  

---

## 🚀 Features

### 🟦 Classical AI Search
- Uses TF-IDF vectorizer
- Computes cosine similarity
- Fast and accurate for simple textual queries

### 🟪 Quantum AI Search
- Quantum embeddings from Pennylane
- Encodes policy text into quantum states
- Useful for complex semantic relationships


### 🔐 Login / Register System (Updated)

The project now uses a **MySQL database** for authentication instead of CSV.

#### ✔ Login Flow
- Users log in using username + password.
- Session cookies expire automatically when the browser closes.
- After server restart, login is required again.

#### ✔ Registration Flow
Users can register through the popup modal.  
Credentials are stored in MySQL using:

```sql
CREATE DATABASE policynav;

USE policynav;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    password VARCHAR(100)
);
```

#### ✔ Session Middleware
Configured as:

```python
app.add_middleware(
    SessionMiddleware,
    secret_key="secret123",
    max_age=None   # session ends when browser closes
)
```

This prevents auto-login after restarting the app.


### 📊 Frontend
- Interactive charts using Chart.js
- Modern UI with HTML + CSS
- Supports both classical & quantum search results

---

## 📁 Project Structure
```
infosys_nlp1/
│── app.py
│── users.csv
│── education_policies.csv
│── poverty_policies.csv
│── quantum_model.pkl
│── policy_vectorizer.pkl
│── policy_tfidf_matrix.pkl
│── requirements.txt
│
├── templates/
│   ├── login.html
│   └── education.html
│
├── static/
│   └── (images, css, assets)
```

---

## 🔧 Installation

### 1️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Run Application
```
uvicorn app:app --reload
```

### 3️⃣ Open in Browser
```
http://127.0.0.1:8000/login
```

---



You can also create new accounts using the Register modal.

---

## 🧪 Example Queries

### Education Queries
- teacher training in rural schools  
- improving learning outcomes  
- digital device distribution to students  

### Poverty Queries
- poverty reduction programs  
- subsidies for low income households  
- skill development for unemployed youth  

---

## 📬 Support
If you need:
- Admin panel  
- Password hashing  
- Enhanced UI  
- Deployment support  

Just ask!

