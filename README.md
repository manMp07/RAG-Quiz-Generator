# 🧠 Enhanced RAG-Based MCQ Quiz Generator with Faculty Authentication & Question Bank

A **Retrieval-Augmented Generation (RAG)** powered MCQ Quiz Generator that creates **high-quality, difficulty-aware multiple-choice questions** directly from user-uploaded PDF notes.  
This system adds **faculty authentication**, a **persistent question bank**, **editable quizzes**, and **custom PDF exports** to the core RAG pipeline.

---

## 🚀 Key Features

- 🔐 **Faculty Authentication** – Register/Login using email & password (MongoDB, bcrypt hashing).
- 📄 Upload any **PDF notes / study material**
- 🧩 Automatic **text chunking with overlap**
- 🔎 **Semantic retrieval** using FAISS vector database (cached in MongoDB cloud)
- 🧠 MCQ generation using **Groq LLM (Llama 3.3 70B)** based strictly on retrieved context
- 🎯 **Difficulty-based questions** (Easy / Medium / Hard) with custom seed queries & few‑shot examples
- 🧪 **Deduplication** using semantic similarity + hashing
- ⚡ **FAISS caching** in MongoDB for fast repeated usage
- 🧾 **Readable chunk & vector files** for transparency
- ✏️ **Editable Quiz** – Change difficulty per question, remove unwanted questions
- 💾 **Save Questions to Database** – Build a personal question bank (grouped by course)
- 📚 **Previous Questions Page** – Browse saved questions by course, select any subset, and generate a **custom PDF quiz** (questions only + separate answer key page)
- 🧹 **Cache management** (clear cached FAISS indexes)

---

## 🧠 Why Use RAG Instead of Direct LLM?

| Direct LLM Approach | This Project (RAG-Based) |
|---------------------|-------------------------|
| Reads entire PDF blindly | Retrieves only relevant chunks |
| High hallucination risk | Context-grounded generation |
| No difficulty control | Explicit Easy / Medium / Hard |
| Repetitive questions | Semantic deduplication |
| No transparency | View chunks & embeddings |
| Recomputes every time | Cached FAISS indexes |

➡️ This project **clearly demonstrates why RAG systems outperform direct LLM usage**, especially in academic and educational settings.

---

## 🏗️ System Architecture

```text
PDF Upload
   ↓
Text Extraction (PyMuPDF)
   ↓
Chunking (900 words + overlap)
   ↓
Embeddings (MiniLM – 384D)
   ↓
FAISS Vector Store (Cached in MongoDB)
   ↓
Seed Query → Semantic Retrieval
   ↓
Retrieved Context
   ↓
Groq LLM (Llama 3.3 70B) + Few‑shot samples
   ↓
Deduplication & Validation
   ↓
Editable Quiz UI → Save to Question Bank
   ↓
Previous Questions → Custom PDF Export
```

---
## 🧱 Tech Stack

- **Frontend / UI:** Streamlit (multi‑page)
- **Authentication:** MongoDB + bcrypt (passlib)
- **PDF Parsing:** PyMuPDF (fitz)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS (cached in MongoDB as binary files)
- **LLM:** Groq API (Llama 3.3 70B) – fast inference
- **Similarity Checking:** Scikit-learn (Cosine Similarity)
- **PDF Generation:** ReportLab
- **Caching & Storage:** MongoDB GridFS (for FAISS & chunks) + local readable files

---

## 📄 PDF Processing Pipeline

1. Extracts raw text from the uploaded PDF
2. Splits text into overlapping chunks (900 words, 50 overlap)
3. Generates 384-dimensional embeddings for each chunk
4. Normalizes and stores embeddings in FAISS
5. Automatically caches the FAISS index + chunks in **MongoDB** (per user)
6. Creates local readable `.txt` files for first 5 chunks and vectors (transparency)

This ensures **performance, explainability, and reproducibility**.

---

## 🎯 Difficulty-Based MCQ Generation

If a course is selected (e.g., "Operating Systems", "DSA"):
- Questions are distributed as:
  - **Easy:** ~33%
  - **Medium:** ~33%
  - **Hard:** remaining

Each difficulty level:
- Uses **custom seed queries** (e.g., "key facts" for Easy, "numerical data" for Hard)
- Uses **sample MCQs as few-shot references** (hardcoded for each course & difficulty)
- Forces the LLM to match **difficulty level and complexity**

This produces **exam-oriented MCQs**, not random questions.

---

## 🔁 Deduplication Strategy

Two-layer deduplication ensures uniqueness:

### 1️⃣ Fingerprint-Based
- MD5 hash of question + options
- Prevents exact duplicates

### 2️⃣ Semantic Similarity
- Cosine similarity on question embeddings (MiniLM)
- Skips questions above similarity threshold (default 0.85)

✅ Result: **No repeated or reworded MCQs**

---

## 🧪 Transparency & Explainability

For every uploaded PDF, the system automatically generates local text files:

- `*_chunks.txt` → First 5 extracted text chunks
- `*_vectors.txt` → First 5 full 384-dimensional embeddings

This makes the RAG pipeline **fully transparent**, ideal for:
- Academic evaluation
- Project defense
- Debugging

---

## ✏️ Editable Quiz & Question Bank

After quiz generation, faculty can:
- **Change difficulty** of any question via dropdown
- **Remove questions** individually
- **Save the final set** to a personal question bank (MongoDB)

The saved questions are stored with:
- Course name
- Question text, options, correct answer, difficulty
- User ID and timestamp

---

## 📚 Previous Questions & Custom PDF Export

A separate **"Previous Questions"** page allows faculty to:
- View all saved questions **grouped by course**
- See each question with its options, correct answer, and difficulty
- **Select any subset of questions** using checkboxes
- Generate a **printable PDF** containing:
  - Only the selected questions (no difficulty labels)
  - A separate **answer key page** at the end

This is perfect for creating custom quizzes, practice tests, or exam papers.

---

## ⚡ Performance Optimization

- PDF hashing avoids duplicate processing
- Cached FAISS index reused automatically (per user, per PDF)
- Embeddings computed only once per PDF
- Manual cache clearing available via UI

---

## 🔐 Environment Setup

Create a secrets file:

```toml
# .streamlit/secrets.toml
MONGODB_URI = "mongodb+srv://<username>:<password>@cluster.mongodb.net/"
DB_NAME = "quiz_app"
GROQ_API_KEY = "your_groq_api_key"
```

Install dependencies:

```bash
pip install streamlit faiss-cpu sentence-transformers pymupdf reportlab scikit-learn openai pymongo passlib
```

Run the application:

```bash
streamlit run app.py
```

---

## 📌 Ideal Use Cases

- Exam preparation platforms
- Faculty question paper generation
- E-learning systems with teacher dashboards
- Academic RAG demonstrations

---

## 🏁 Conclusion

This project showcases a **production-grade Retrieval-Augmented Generation system** combining:

- Information Retrieval (FAISS)
- NLP Embeddings
- Large Language Models (Groq)
- Deduplication & Evaluation
- Faculty Authentication & Question Banking
- Custom PDF Export

It clearly answers the question:  
> *“Why not just upload the PDF to an LLM?”*

⭐ If you find this project useful, consider starring the repository!
