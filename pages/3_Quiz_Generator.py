import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import random
import re
import io
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import pickle
from datetime import datetime

from utils.auth import logout
from utils.storage import save_faiss_to_mongodb, load_faiss_from_mongodb
from utils.db import save_questions, get_db

# ------------------------------
# Authentication check
# ------------------------------
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to access the quiz generator.")
    st.page_link("pages/1_Login.py", label="Go to Login")
    st.stop()

st.set_page_config(page_title="Quiz Generator", page_icon="sparkles.png", layout="wide")
st.title("Enhanced RAG Quiz Generator with Difficulty Levels")
st.caption(f"Logged in as: {st.session_state['user_email']}")

# Logout button in sidebar
with st.sidebar:
    if st.button("Logout"):
        logout()
        st.switch_page("pages/1_Login.py")

# ------------------------------
# All the original helper functions (unchanged except where noted)
# ------------------------------

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def chunk_text(text, chunk_size=900, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    chunks = [c.strip() for c in chunks if len(c.strip()) > 0]
    return chunks

def create_readable_chunks_file(pdf_hash, chunks, storage_dir="faiss_storage"):
    """Create readable text file with 5 chunks (locally)"""
    os.makedirs(storage_dir, exist_ok=True)
    chunks_file = f"{storage_dir}/{pdf_hash}_chunks.txt"
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write("TEXT CHUNKS EXTRACTED FROM PDF\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total chunks in PDF: {len(chunks)}\n")
        f.write(f"Size of each chunk: {len(chunks[0].split())}\n")
        f.write(f"Showing first 5 chunks below:\n\n")
        for i in range(min(5, len(chunks))):
            f.write(f"CHUNK {i+1}:\n")
            f.write(f"Character Count: {len(chunks[i])}\n")
            f.write(chunks[i] + "\n\n")
    return chunks_file

def create_readable_vectors_file(pdf_hash, embeddings, storage_dir="faiss_storage"):
    os.makedirs(storage_dir, exist_ok=True)
    vectors_file = f"{storage_dir}/{pdf_hash}_vectors.txt"
    with open(vectors_file, "w", encoding="utf-8") as f:
        f.write("VECTOR EMBEDDINGS (FULL 384 DIMENSIONS)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total vectors: {len(embeddings)}\n")
        f.write(f"Dimensions per vector: {embeddings.shape[1]}\n")
        f.write(f"Showing first 5 vectors with all 384 dimensions:\n\n")
        for i in range(min(5, len(embeddings))):
            f.write(f"VECTOR {i+1} (All 384 dimensions):\n")
            f.write("[ ")
            for j, value in enumerate(embeddings[i]):
                f.write(f"{value:.6f} ")
            f.write("]\n\n")
    return vectors_file

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index(chunks, embedder, pdf_hash=None):
    """Build FAISS index and optionally save to GCS"""
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    
    if pdf_hash:
        # Save to MongoDB
        save_faiss_to_mongodb(pdf_hash, index, chunks, st.session_state["user_id"])
        # Also create readable text files locally
        create_readable_chunks_file(pdf_hash, chunks)
        create_readable_vectors_file(pdf_hash, embeddings)
    
    return index, embeddings

def search_chunks(query, chunks, index, embedder, top_k=4):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    return [chunks[i] for i in I[0]]

def get_pdf_hash(pdf_file):
    pdf_file.seek(0)
    pdf_content = pdf_file.read()
    pdf_file.seek(0)
    return hashlib.md5(pdf_content).hexdigest()

# ------------------------------
# Sample questions and seed queries
# ------------------------------
SAMPLE_QUESTIONS = {
    "Operating Systems": {
        "Easy": [
            {
                "question": "The operating system is responsible for?",
                "options": {
                    "a": "bad-block recovery",
                    "b": "booting from disk", 
                    "c": "disk initialization",
                    "d": "all of the mentioned"
                }
            },
            {
                "question": "What is the central, fundamental component of an operating system that manages the system's resources and acts as the bridge between hardware and application software?",
                "options": {
                    "a": "Shell",
                    "b": "Compiler", 
                    "c": "Kernel",
                    "d": "BIOS"
                }
            },
            {
                "question": "Which part of the Operating System allows users to interact with the computer, either through commands or a graphical interface?",
                "options": {
                    "a": "Kernel",
                    "b": "CPU", 
                    "c": "Shell / User Interface (UI)",
                    "d": "Device Driver"
                }
            },
            {
                "question": "The Round Robin (RR) scheduling algorithm relies on a small, fixed unit of time given to each process before it is interrupted. What is this time unit called?",
                "options": {
                    "a": "Burst Time",
                    "b": "Turnaround Time", 
                    "c": "Time Quantum",
                    "d": "Wait Time"
                }
            },
            {
                "question": "What are the two fundamental, atomic operations used to manipulate the value of a semaphore?",
                "options": {
                    "a": "Load and Store",
                    "b": "Read and Write", 
                    "c": "Wait (or P) and Signal (or V)",
                    "d": "Lock and Unlock"
                }
            }
        ],
        "Medium": [
            {
                "question": "For an effective operating system, when to check for deadlock?",
                "options": {
                    "a": "every time a resource request is made at fixed time intervals",
                    "b": "at fixed time intervals", 
                    "c": "every time a resource request is made",
                    "d": "none of the mentioned"
                }
            },
            {
                "question": "Which of the four necessary conditions for deadlock is broken by implementing the Banker's Algorithm for resource allocation?",
                "options": {
                    "a": "Mutual Exclusion",
                    "b": "Hold and Wait", 
                    "c": "No Preemption",
                    "d": "Circular Wait"
                }
            },
            {
                "question": "In the Round Robin (RR) CPU scheduling algorithm, what is the effect of setting a very small time quantum?",
                "options": {
                    "a": "It decreases context switching overhead and increases throughput.",
                    "b": "It makes the algorithm behave similarly to First-Come, First-Served (FCFS).", 
                    "c": "It increases context switching overhead, but improves response time for short jobs.",
                    "d": "It leads to internal fragmentation in the main memory."
                }
            },
            {
                "question": "What is the fundamental difference between a Thread and a Process in terms of memory space?",
                "options": {
                    "a": "Processes have individual stacks, while threads share a single common stack.",
                    "b": "Threads within the same process share the same memory space, while processes have separate memory spaces.", 
                    "c": "Processes can only run on a single CPU core, but threads can run on multiple cores.",
                    "d": "Threads are managed by the operating system, but processes are managed by the user application."
                }
            },
            {
                "question": "If a Counting Semaphore is initialized to 8, what does this initial value specifically signify to the operating system?",
                "options": {
                    "a": "The maximum number of processes allowed in the Ready queue.",
                    "b": "The number of available instances of the shared resource(s).", 
                    "c": "The maximum number of processes allowed to wait for the resource.",
                    "d": "The priority level assigned to the semaphore."
                }
            }
        ],
        "Hard": [
            {
                "question": "A counting semaphore S is initialized to a value of 7. A set of processes issues the following operations on S in a specific order: 10 P (wait) operations and 4 V (signal) operations. Assuming P operations decrement the semaphore and block if the value is negative, and V operations increment it, how many processes will be in the blocked queue after all operations are completed?",
                "options": {
                    "a": "0",
                    "b": "1", 
                    "c": "3",
                    "d": "2"
                }
            },
            {
                "question": "A file system uses a block size of 4 KB and a block pointer size of 4 bytes. An inode contains 10 direct pointers and 1 single-indirect pointer. What is the maximum file size that can be addressed using this structure? (Note: 1 KB = 1024 bytes)",
                "options": {
                    "a": "4.04 MB",
                    "b": "1.04 MB", 
                    "c": "40 KB",
                    "d": "2.04 MB"
                }
            },
            {
                "question": "A set of processes arrive with the following arrival times and burst times. The system uses Round Robin scheduling with a time quantum (TQ) of 2 ms. Process | Arrival Time | Burst Time P1 | 0 ms | 5 ms P2 | 1 ms | 3 ms P3 | 2 ms | 4 ms P4 | 4 ms | 2 ms What is the average waiting time for these processes?",
                "options": {
                    "a": "5.50 ms",
                    "b": "7.25 ms", 
                    "c": "8.00 ms",
                    "d": "6.75 ms"
                }
            },
            {
                "question": "A system with 4 page frames uses the First-In, First-Out (FIFO) page replacement algorithm. The system accesses pages in the following order (reference string): 7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2 How many page faults will occur?",
                "options": {
                    "a": "6",
                    "b": "7", 
                    "c": "8",
                    "d": "9"
                }
            },
            {
                "question": "A disk drive has 200 cylinders, numbered 0 to 199. The disk head is currently at cylinder 53, and the head is moving towards the higher-numbered cylinders (towards 199). The queue of pending requests, in order of arrival, is: 98, 183, 37, 122, 14, 124, 65 Using the C-SCAN (Circular-SCAN) algorithm, what is the total head movement in cylinders?",
                "options": {
                    "a": "322",
                    "b": "299", 
                    "c": "382",
                    "d": "345"
                }
            }
        ]
    }
}

SEED_QUERIES = {
    "Easy": [
        "key facts and important information",
        "main conclusions and summaries", 
        "important terminology and meanings",
        "dates, timelines, and chronological information",
        "key concepts and definitions in the document",
        "significant findings and results",
        "real-world applications and examples",
        "use cases and scenarios described",
        "basic principles and fundamental ideas",
        "simple processes and procedures"
    ],
    
    "Medium": [
        "fundamental principles explained in this document",
        "step-by-step processes and procedures", 
        "sequences and order of operations",
        "methodologies and approaches described",
        "comparisons and contrasts between concepts",
        "advantages and disadvantages discussed",
        "cause and effect relationships described",
        "problems and solutions discussed",
        "key concepts and definitions in the document",
        "real-world applications and examples"
    ],
    
    "Hard": [
        "numerical data, statistics, and percentages",
        "measurements, quantities, and mathematical information",
        "strengths and weaknesses mentioned",
        "implications and consequences",
        "case studies and practical implementations",
        "complex relationships between concepts",
        "theoretical foundations and frameworks",
        "limitations and constraints discussed",
        "advanced methodologies and techniques",
        "critical analysis and evaluation points"
    ]
}

# ------------------------------
# Deduplication functions
# ------------------------------
def calculate_question_similarity(question1, question2, embedder):
    emb1 = embedder.encode([question1])
    emb2 = embedder.encode([question2])
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def is_similar_question(new_question, existing_questions, embedder, similarity_threshold=0.85):
    for existing in existing_questions:
        similarity = calculate_question_similarity(new_question, existing, embedder)
        if similarity > similarity_threshold:
            return True
    return False

def get_question_fingerprint(mcq_data):
    content = mcq_data['question'] + ''.join(mcq_data['options'])
    return hashlib.md5(content.encode()).hexdigest()

# ------------------------------
# LLM: Groq
# ------------------------------
@st.cache_resource
def get_groq_client():
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )

def get_llm_response(prompt, model="llama-3.3-70b-versatile", max_retries=3):
    client = get_groq_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert quiz generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"Failed to get response from Groq after {max_retries} attempts.")
                return None


def prompt_mcq_with_difficulty(context, query_type, difficulty, sample_questions):
    """Enhanced prompt with difficulty level and sample questions"""
    
    # Format sample questions for the prompt
    sample_text = ""
    for i, sample in enumerate(sample_questions, 1):
        sample_text += f"SAMPLE {i}:\n"
        sample_text += f"Question: {sample['question']}\n"
        sample_text += "Options:\n"
        for opt_key, opt_value in sample['options'].items():
            sample_text += f"  {opt_key}) {opt_value}\n"
        sample_text += "\n"
    
    prompt = f"""
You are an expert exam question setter. Based ONLY on the given context, write ONE multiple-choice question.
The question should focus on: {query_type}
The difficulty level should be: {difficulty}

IMPORTANT: Create a question with the SAME DIFFICULTY LEVEL as the sample questions below. 
The question should NOT be exactly the same as the samples, but should have similar complexity and test similar concepts.

FOLLOW THIS EXACT OUTPUT FORMAT:

Q: [Your question here?]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]

SAMPLE QUESTIONS FOR REFERENCE (Difficulty: {difficulty}):
{sample_text}

Context:
{context}

Remember:
1. Create questions that test understanding at the {difficulty.lower()} level
2. Make options plausible but distinct
3. Ensure the question is clear and unambiguous  
4. Base everything ONLY on the provided context
5. Match the complexity and depth of the sample questions
6. Use the exact output format shown above
"""
    return prompt.strip()

def prompt_mcq(context, query_type):
    """Original prompt for non-difficulty based generation"""
    prompt = f"""
You are an expert exam question setter. Based ONLY on the given context, write ONE multiple-choice question.
The question should focus on: {query_type}

Follow the output format EXACTLY:

---
EXAMPLE

Context: The first part of the OS to load on start-up is the kernel.
Q: What is the first part of the operating system to load when a computer starts?
A) The Shell
B) The Kernel
C) Application Software
D) The GUI
Correct: B
---

IMPORTANT INSTRUCTIONS:
1. Create questions that test understanding, not just recall
2. Make options plausible but distinct
3. Ensure the question is clear and unambiguous
4. Base everything ONLY on the provided context

Context:
{context}
"""
    return prompt.strip()

def parse_mcq(text):
    t = text.strip()
    m_q = re.search(r"Q:\s*(.+)", t, re.IGNORECASE | re.DOTALL)
    if not m_q:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        q = lines[0] if lines else "Generated question"
    else:
        q = m_q.group(1).strip()
        q = q.split("\nA)")[0].strip()

    def find_opt(letter):
        m = re.search(rf"{letter}\)\s*(.+)", t, re.IGNORECASE)
        return m.group(1).strip() if m else f"Option {letter}"

    A = find_opt("A")
    B = find_opt("B")
    C = find_opt("C")
    D = find_opt("D")

    m_corr = re.search(r"Correct:\s*([ABCD])", t, re.IGNORECASE)
    correct = m_corr.group(1).upper() if m_corr else "A"

    def trim(s, n=300):
        return (s[:n] + "…") if len(s) > n else s

    return {
        "question": trim(q, 500),
        "options": [trim(A), trim(B), trim(C), trim(D)],
        "answer": correct,
        "query_type": "unknown",  # Will be set by caller
        "difficulty": "unknown"   # Will be set by caller
    }


# ------------------------------
# Enhanced Quiz Generation
# ------------------------------
def generate_diverse_quiz(num_questions, chunks, index, embedder, top_k=2, selected_course=None):
    """Generate quiz with deduplication and diversity"""
    quiz = []
    used_fingerprints = set()
    used_query_types = set()
    max_attempts_per_question = 5
    total_max_attempts = num_questions * 10
    
    # If course is selected, distribute questions across difficulties
    if selected_course and selected_course in SAMPLE_QUESTIONS:
        # Calculate distribution
        easy_count = num_questions // 3
        medium_count = num_questions // 3
        hard_count = num_questions - easy_count - medium_count
        
        st.write(f"**Difficulty Distribution:** Easy: {easy_count}, Medium: {medium_count}, Hard: {hard_count}")
        
        # Generate questions for each difficulty
        for difficulty, count in [("Easy", easy_count), ("Medium", medium_count), ("Hard", hard_count)]:
            if count > 0:
                st.write(f"Generating {count} {difficulty} questions...")
                difficulty_quiz = generate_difficulty_quiz(
                    difficulty, count, chunks, index, embedder, top_k, 
                    selected_course, used_fingerprints, used_query_types
                )
                quiz.extend(difficulty_quiz)
                used_fingerprints.update([get_question_fingerprint(q) for q in difficulty_quiz])
    else:
        # Original behavior - no difficulty level
        st.write("Generating questions without difficulty levels...")
        attempts = 0
        while len(quiz) < num_questions and attempts < total_max_attempts:
            attempts += 1
            
            # Use all available queries
            available_queries = [q for q in SEED_QUERIES["Easy"] + SEED_QUERIES["Medium"] + SEED_QUERIES["Hard"] 
                               if q not in used_query_types]
            if not available_queries:
                available_queries = SEED_QUERIES["Easy"] + SEED_QUERIES["Medium"] + SEED_QUERIES["Hard"]
                
            query = random.choice(available_queries)
            query_type = query
            
            try:
                contexts = search_chunks(query, chunks, index, embedder, top_k=top_k)
                context_text = "\n\n".join(contexts)
                
                if len(context_text.strip()) < 50:  # Skip if context is too small
                    continue
                    
                full_prompt = prompt_mcq(context_text, query_type)
                raw_output = get_llm_response(full_prompt)
                
                if not raw_output:
                    continue
                    
                if "Q:" in raw_output and "A)" in raw_output and "Correct:" in raw_output:
                    mcq = parse_mcq(raw_output)
                    mcq["query_type"] = query_type
                    mcq["difficulty"] = "Not Specified"
                    
                    # Deduplication checks
                    fingerprint = get_question_fingerprint(mcq)
                    
                    if (fingerprint not in used_fingerprints and 
                        not is_similar_question(mcq['question'], 
                                              [q['question'] for q in quiz], 
                                              embedder)):
                        
                        quiz.append(mcq)
                        used_fingerprints.add(fingerprint)
                        used_query_types.add(query_type)
                        st.write(f"✅ Generated question {len(quiz)}: {query_type}")
                    else:
                        st.write(f"🔄 Skipped similar question: {query_type}")
                else:
                    continue
                    
            except Exception as e:
                st.write(f"❌ Error generating question: {str(e)}")
                continue
    
    return quiz

def generate_difficulty_quiz(difficulty, num_questions, chunks, index, embedder, top_k, 
                           selected_course, used_fingerprints, used_query_types):
    """Generate questions for a specific difficulty level"""
    quiz = []
    max_attempts = num_questions * 5
    
    attempts = 0
    while len(quiz) < num_questions and attempts < max_attempts:
        attempts += 1
        
        # Use queries specific to the difficulty level
        available_queries = [q for q in SEED_QUERIES[difficulty] if q not in used_query_types]
        if not available_queries:
            available_queries = SEED_QUERIES[difficulty]
            
        query = random.choice(available_queries)
        query_type = query
        
        try:
            contexts = search_chunks(query, chunks, index, embedder, top_k=top_k)
            context_text = "\n\n".join(contexts)
            
            if len(context_text.strip()) < 50:  # Skip if context is too small
                continue
                
            # Get sample questions for this difficulty and course
            sample_questions = SAMPLE_QUESTIONS[selected_course][difficulty][:3]  # Use first 3 as samples
            
            full_prompt = prompt_mcq_with_difficulty(context_text, query_type, difficulty, sample_questions)
            raw_output = get_llm_response(full_prompt)
            
            if not raw_output:
                continue
                
            if "Q:" in raw_output and "A)" in raw_output and "Correct:" in raw_output:
                mcq = parse_mcq(raw_output)
                mcq["query_type"] = query_type
                mcq["difficulty"] = difficulty
                
                # Deduplication checks
                fingerprint = get_question_fingerprint(mcq)
                
                if (fingerprint not in used_fingerprints and 
                    not is_similar_question(mcq['question'], 
                                          [q['question'] for q in quiz], 
                                          embedder)):
                    
                    quiz.append(mcq)
                    used_fingerprints.add(fingerprint)
                    used_query_types.add(query_type)
                    st.write(f"✅ Generated {difficulty} question {len(quiz)}: {query_type}")
                else:
                    st.write(f"🔄 Skipped similar {difficulty} question: {query_type}")
            else:
                continue
                
        except Exception as e:
            st.write(f"❌ Error generating {difficulty} question: {str(e)}")
            continue
    
    return quiz

# ------------------------------
# PDF export for selected questions (new)
# ------------------------------
def make_selected_questions_pdf(questions, title="Custom Quiz"):
    """Generate PDF for selected questions (no difficulties, answer key at end)."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    x = margin
    y = height - margin

    def draw_wrapped(text, y, leading=14, indent=0):
        max_width = width - 2*margin - indent
        wrapped = textwrap.wrap(text, width=95)
        for line in wrapped:
            if y < margin + 2*cm:
                c.showPage()
                y = height - margin
            c.drawString(x + indent, y, line)
            y -= leading
        return y

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 24
    c.setFont("Helvetica", 11)

    # Questions
    for i, q in enumerate(questions, 1):
        y = draw_wrapped(f"{i}. {q['question_text']}", y, leading=14)
        for label, opt in zip(["A", "B", "C", "D"], q["options"]):
            y = draw_wrapped(f"   {label}) {opt}", y, leading=14)
        y -= 8
        if y < margin + 2*cm:
            c.showPage()
            y = height - margin

    # Answer Key (new page)
    c.showPage()
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Answer Key")
    y -= 20
    c.setFont("Helvetica", 11)
    for i, q in enumerate(questions, 1):
        answer_letter = q["correct_answer"]
        answer_text = f"{i}. {answer_letter}"
        y = draw_wrapped(answer_text, y, leading=14)
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------------
# Sidebar UI (adapted for editing)
# ------------------------------
with st.sidebar:
    st.header("Upload & Settings")
    pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    available_courses = ["None"] + list(SAMPLE_QUESTIONS.keys())
    selected_course = st.selectbox(
        "Select Course (for difficulty-based questions)",
        available_courses,
        help="Select 'None' for general quiz generation without difficulty levels"
    )
    if selected_course != "None":
        st.caption("📝 Make sure you've uploaded the same course PDF")
    
    num_q = st.number_input("Number of questions", min_value=1, max_value=50, value=10, step=1)
    top_k = st.slider("Retriever top-k passages per question", min_value=1, max_value=8, value=2)
    similarity_threshold = st.slider("Similarity threshold for deduplication",
                                     min_value=0.5, max_value=0.95, value=0.85, step=0.05)
    
    if selected_course != "None":
        easy_count = num_q // 3
        medium_count = num_q // 3
        hard_count = num_q - easy_count - medium_count
        st.info(f"**Difficulty Distribution:**\n- Easy: {easy_count}\n- Medium: {medium_count}\n- Hard: {hard_count}")
    
    st.caption("Stack: PyMuPDF + MiniLM + FAISS + Groq + ReportLab")
    
    if st.button("Build Quiz"):
        if not pdf_file:
            st.error("Please upload a PDF first.")
        else:
            pdf_hash = get_pdf_hash(pdf_file)
            
            with st.spinner("Checking for cached index in MongoDB..."):
                cached_index, cached_chunks = load_faiss_from_mongodb(
                    pdf_hash, st.session_state["user_id"]
                )
            
            if cached_index and cached_chunks:
                st.success("✅ Loaded from MongoDB cache! Generating questions...")
                index = cached_index
                chunks = cached_chunks
                embedder = load_embedding_model()
                embeddings = embedder.encode(chunks, convert_to_numpy=True)
                create_readable_chunks_file(pdf_hash, chunks)
                create_readable_vectors_file(pdf_hash, embeddings)
            else:
                with st.spinner("Extracting PDF text..."):
                    text = extract_text_from_pdf(pdf_file)
                
                if len(text.strip()) < 50:
                    st.error("Could not extract enough text from the PDF.")
                else:
                    with st.spinner("Chunking + embedding + indexing..."):
                        chunks = chunk_text(text)
                        embedder = load_embedding_model()
                        index, embeddings = build_faiss_index(chunks, embedder, pdf_hash)
                        st.info("📦 New index built and cached in MongoDB.")
            
            if 'index' in locals() and 'chunks' in locals():
                with st.spinner("Generating questions with Groq..."):
                    course_for_generation = selected_course if selected_course != "None" else None
                    quiz = generate_diverse_quiz(int(num_q), chunks, index, embedder, top_k, course_for_generation)
                    # Store in session state for editing
                    st.session_state["editable_quiz"] = quiz
                    st.session_state["current_pdf_hash"] = pdf_hash
                    st.session_state["current_course"] = selected_course
                    if quiz:
                        st.success(f"✅ Generated {len(quiz)} unique questions! You can now edit below.")
                    else:
                        st.error("Failed to generate any valid questions. Try with different settings.")
    
    st.markdown("---")
    st.subheader("Cache Management")
    if st.button("Clear My Cached Indexes"):
        st.warning("Clearing cache is not implemented yet. You can delete entries manually from MongoDB.")

# ------------------------------
# Editable Quiz Display
# ------------------------------
if "editable_quiz" in st.session_state and st.session_state["editable_quiz"]:
    st.subheader("📝 Edit Quiz")
    st.caption("Modify difficulties, remove questions, and save to your question bank.")

    quiz = st.session_state["editable_quiz"]
    updated_quiz = []
    
    for idx, q in enumerate(quiz):
        with st.container(border=True):
            cols = st.columns([0.8, 0.2])
            with cols[0]:
                st.markdown(f"**Q{idx+1}: {q['question']}**")
                st.markdown(f"A) {q['options'][0]}")
                st.markdown(f"B) {q['options'][1]}")
                st.markdown(f"C) {q['options'][2]}")
                st.markdown(f"D) {q['options'][3]}")
                st.markdown(f"*Correct Answer: {q['answer']}*")
            with cols[1]:
                # Difficulty dropdown
                current_diff = q.get("difficulty", "Not Specified")
                new_diff = st.selectbox(
                    "Difficulty",
                    options=["Easy", "Medium", "Hard", "Not Specified"],
                    index=["Easy", "Medium", "Hard", "Not Specified"].index(current_diff),
                    key=f"diff_{idx}"
                )
                # Remove button
                if st.button("🗑️ Remove", key=f"remove_{idx}"):
                    st.session_state[f"remove_{idx}"] = True
                    st.rerun()
                # Store updated difficulty
                q["difficulty"] = new_diff
            
            # Check if this question should be removed
            if not st.session_state.get(f"remove_{idx}", False):
                updated_quiz.append(q)
            else:
                # Clear the removal flag for next render
                st.session_state.pop(f"remove_{idx}", None)
    
    # Update session state with modified quiz
    st.session_state["editable_quiz"] = updated_quiz

    # Save button
    if st.button("💾 Save Quiz to Question Bank"):
        if not updated_quiz:
            st.warning("No questions to save.")
        else:
            try:
                save_questions(
                    user_id=st.session_state["user_id"],
                    course=st.session_state.get("current_course", "None"),
                    questions=updated_quiz
                )
                st.success(f"✅ Saved {len(updated_quiz)} questions to your question bank!")
                # Optionally clear the editable quiz after save
                # st.session_state["editable_quiz"] = []
            except Exception as e:
                st.error(f"Error saving questions: {e}")

    # Optional: Preview button to see current list
    with st.expander("Preview current questions"):
        for i, q in enumerate(updated_quiz):
            st.write(f"{i+1}. {q['question']} [{q['difficulty']}]")
else:
    st.info("Upload a PDF and click **Build Quiz** from the sidebar to get started.")