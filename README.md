# 🧠 Enhanced RAG-Based MCQ Quiz Generator

A **Retrieval-Augmented Generation (RAG)** powered MCQ Quiz Generator that creates **high-quality, difficulty-aware multiple-choice questions** directly from user-uploaded PDF notes.

This project goes beyond simple *PDF → LLM* approaches by using **semantic retrieval, vector databases, deduplication, caching, and controlled prompt engineering** to generate **accurate, diverse, and exam-level MCQs**.

---

## 🚀 Key Features

- 📄 Upload any **PDF notes / study material**
- 🧩 Automatic **text chunking with overlap**
- 🔎 **Semantic retrieval** using FAISS vector database
- 🧠 MCQ generation using **Gemini LLM** based strictly on retrieved context
- 🎯 **Difficulty-based questions** (Easy / Medium / Hard)
- 🧪 **Deduplication** using semantic similarity + hashing
- ⚡ **FAISS caching** for fast repeated usage
- 🧾 **Readable chunk & vector files** for transparency
- 📊 Interactive quiz attempt with scoring
- 📄 Export quiz as **PDF with answer key**
- 🧹 Cache management from UI

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
FAISS Vector Store (Cached)
   ↓
Seed Query → Semantic Retrieval
   ↓
Retrieved Context
   ↓
Gemini LLM (MCQ Generation)
   ↓
Deduplication & Validation
   ↓
Quiz UI + PDF Export
```

---

## 🧱 Tech Stack

- **Frontend / UI:** Streamlit
- **PDF Parsing:** PyMuPDF (fitz)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS
- **LLM:** Google Gemini (gemini-2.0-flash)
- **Similarity Checking:** Scikit-learn (Cosine Similarity)
- **PDF Generation:** ReportLab
- **Caching & Storage:** Local Disk (FAISS + Pickle)

---

## 📄 PDF Processing Pipeline

1. Extracts raw text from the uploaded PDF
2. Splits text into overlapping chunks (900 words)
3. Generates 384-dimensional embeddings for each chunk
4. Normalizes and stores embeddings in FAISS
5. Automatically saves:
   - FAISS index (`.faiss`)
   - Chunk data (`.pkl`)
   - Readable text chunks (`_chunks.txt`)
   - Readable vector embeddings (`_vectors.txt`)

This ensures **performance, explainability, and reproducibility**.

---

## 🎯 Difficulty-Based MCQ Generation

If a course is selected:
- Questions are distributed as:
  - **Easy:** ~33%
  - **Medium:** ~33%
  - **Hard:** Remaining

Each difficulty level:
- Uses **custom seed queries**
- Uses **sample MCQs as few-shot references**
- Forces Gemini to match **difficulty level and complexity**

This produces **exam-oriented MCQs**, not random questions.

---

## 🔁 Deduplication Strategy

Two-layer deduplication ensures uniqueness:

### 1️⃣ Fingerprint-Based
- MD5 hash of question + options
- Prevents exact duplicates

### 2️⃣ Semantic Similarity
- Cosine similarity on question embeddings
- Skips questions above similarity threshold

✅ Result: **No repeated or reworded MCQs**

---

## 🧪 Transparency & Explainability

For every uploaded PDF, the system automatically generates:

- `*_chunks.txt` → First 5 extracted text chunks
- `*_vectors.txt` → First 5 full 384-dimensional embeddings

This makes the RAG pipeline **fully transparent**, ideal for:
- Academic evaluation
- Project defense
- Debugging

---

## 📊 Quiz Attempt & Evaluation

- Interactive quiz interface
- Difficulty tags per question
- Overall score calculation
- Difficulty-wise performance analysis
- Detailed feedback for each question

---

## 📄 PDF Export

- Generates a professional PDF containing:
  - MCQ questions
  - Difficulty indicators
  - Complete answer key
- Suitable for:
  - Offline exams
  - Faculty review
  - Sharing with students

---

## ⚡ Performance Optimization

- PDF hashing avoids duplicate processing
- Cached FAISS index reused automatically
- Embeddings computed only once per PDF
- Manual cache clearing available via UI

---

## 🔐 Environment Setup

Create a secrets file:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_api_key_here"
```

Install dependencies:

```bash
pip install streamlit faiss-cpu sentence-transformers pymupdf reportlab scikit-learn google-generativeai
```

Run the application:

```bash
streamlit run app.py
```

---

## 📌 Ideal Use Cases

- Exam preparation platforms
- Faculty question paper generation
- E-learning systems
- Academic RAG demonstrations
- NLP + IR course projects

---

## 🏁 Conclusion

This project showcases a **production-grade Retrieval-Augmented Generation system** combining:

- Information Retrieval (FAISS)
- NLP Embeddings
- Large Language Models
- Deduplication & Evaluation
- PDF Export & UI

It clearly answers the question:
> *“Why not just upload the PDF to an LLM?”*

⭐ If you find this project useful, consider starring the repository!
