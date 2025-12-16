import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
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

# ------------------------------
# Helpers: PDF text extraction
# ------------------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

# ------------------------------
# Helpers: Chunking
# ------------------------------
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

# ------------------------------
# Readable Text File Creation
# ------------------------------
def create_readable_chunks_file(pdf_hash, chunks, storage_dir="faiss_storage"): 
    """Create readable text file with 5 chunks"""
    chunks_file = f"{storage_dir}/{pdf_hash}_chunks.txt"
    
    with open(chunks_file, "w", encoding="utf-8") as f:
        f.write("TEXT CHUNKS EXTRACTED FROM PDF\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total chunks in PDF: {len(chunks)}\n")
        f.write(f"Size of each chunk: {len(chunks[0].split())}\n")
        f.write(f"Showing first 5 chunks below:\n\n")
        
        # Write first 5 chunks (full 900-word chunks)
        for i in range(min(5, len(chunks))):
            f.write(f"CHUNK {i+1}:\n")
            f.write(f"Character Count: {len(chunks[i])}\n")
            f.write(chunks[i] + "\n\n")
    
    print(f"Created readable chunks file: {chunks_file}")
    return chunks_file

def create_readable_vectors_file(pdf_hash, embeddings, storage_dir="faiss_storage"):
    """Create readable text file with 5 full vectors"""
    vectors_file = f"{storage_dir}/{pdf_hash}_vectors.txt"
    
    with open(vectors_file, "w", encoding="utf-8") as f:
        f.write("VECTOR EMBEDDINGS (FULL 384 DIMENSIONS)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total vectors: {len(embeddings)}\n")
        f.write(f"Dimensions per vector: {embeddings.shape[1]}\n")
        f.write(f"Showing first 5 vectors with all 384 dimensions:\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Write first 5 vectors (all 384 dimensions)
        for i in range(min(5, len(embeddings))):
            f.write(f"VECTOR {i+1} (All 384 dimensions):\n")
            
            # Write all 384 dimensions
            f.write("[ ")
            for j, value in enumerate(embeddings[i]):
                f.write(f"{value:.6f} ")
            f.write("]\n\n")
    
    print(f"✅ Created readable vectors file: {vectors_file}")
    return vectors_file

# ------------------------------
# Embeddings + FAISS
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index(chunks, embedder, pdf_hash=None):
    """Build FAISS index and automatically create readable text files"""
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    
    # Save to disk if pdf_hash is provided
    if pdf_hash:
        storage_dir = ensure_storage_dir()
        
        # Save FAISS binary files
        save_faiss_to_disk(pdf_hash, index, chunks, storage_dir)
        
        # ✅ AUTOMATICALLY CREATE READABLE TEXT FILES
        create_readable_chunks_file(pdf_hash, chunks, storage_dir)
        create_readable_vectors_file(pdf_hash, embeddings, storage_dir)
    
    return index, embeddings

def search_chunks(query, chunks, index, embedder, top_k=4):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# Creating Storage Directory
# -----------------------------
def ensure_storage_dir():
    """Create directory for storing FAISS files if it doesn't exist"""
    os.makedirs("faiss_storage", exist_ok=True)
    return "faiss_storage"

# ------------------------------
# FAISS File Storage Functions
# ------------------------------
def get_pdf_hash(pdf_file):
    """Generate unique hash for PDF file"""
    pdf_file.seek(0)  # Reset file pointer
    pdf_content = pdf_file.read()
    pdf_file.seek(0)  # Reset again for future use
    return hashlib.md5(pdf_content).hexdigest()

def save_faiss_to_disk(pdf_hash, index, chunks, storage_dir="faiss_storage"):
    """Save FAISS index and chunks to disk"""
    # Save FAISS index
    faiss.write_index(index, f"{storage_dir}/{pdf_hash}.faiss")
    
    # Save chunks
    with open(f"{storage_dir}/{pdf_hash}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"✅ Saved FAISS index for PDF: {pdf_hash}")

def load_faiss_from_disk(pdf_hash, storage_dir="faiss_storage"):
    """Load FAISS index and chunks from disk if they exist"""
    index_path = f"{storage_dir}/{pdf_hash}.faiss"
    chunks_path = f"{storage_dir}/{pdf_hash}_chunks.pkl"
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load chunks
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            
            print(f"✅ Loaded cached FAISS index for PDF: {pdf_hash}")
            return index, chunks
        except Exception as e:
            print(f"❌ Error loading cached index: {e}")
            return None, None
    
    return None, None

def is_faiss_cached(pdf_hash, storage_dir="faiss_storage"):
    """Check if FAISS index exists for this PDF"""
    return os.path.exists(f"{storage_dir}/{pdf_hash}.faiss")

# ------------------------------
# Sample Questions for Difficulty Levels
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

# ------------------------------
# Seed Queries for Each Difficulty Level
# ------------------------------
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
# Deduplication Functions
# ------------------------------
def calculate_question_similarity(question1, question2, embedder):
    """Calculate cosine similarity between two questions"""
    emb1 = embedder.encode([question1])
    emb2 = embedder.encode([question2])
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def is_similar_question(new_question, existing_questions, embedder, similarity_threshold=0.85):
    """Check if new question is too similar to existing ones"""
    for existing in existing_questions:
        similarity = calculate_question_similarity(new_question, existing, embedder)
        if similarity > similarity_threshold:
            return True
    return False

def get_question_fingerprint(mcq_data):
    """Create a simple fingerprint of the question content"""
    content = mcq_data['question'] + ''.join(mcq_data['options'])
    return hashlib.md5(content.encode()).hexdigest()

# ------------------------------
# LLM: Gemini Pro
# ------------------------------
def get_gemini_response(prompt):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"--- GEMINI API ERROR ---")
        print(e)
        print(f"------------------------")
        st.error(f"An error occurred with the Gemini API. Check terminal for details.")
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
                raw_output = get_gemini_response(full_prompt)
                
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
            raw_output = get_gemini_response(full_prompt)
            
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

# Pdf Convert
def make_quiz_pdf(quiz, title="Generated MCQ Quiz"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    x = margin
    y = height - margin

    def draw_wrapped(text, y, leading=14, indent=0):
        max_width = width - 2*margin - indent
        wrapped = []
        for line in text.split("\n"):
            wrapped += textwrap.wrap(line, width=95)
        for line in wrapped:
            if y < margin + 2*cm:
                c.showPage()
                y = height - margin
            c.drawString(x + indent, y, line)
            y -= leading
        return y

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 24
    c.setFont("Helvetica", 11)
    
    # Add difficulty information if available
    difficulties = [q.get('difficulty', 'Not Specified') for q in quiz]
    if any(d != 'Not Specified' for d in difficulties):
        difficulty_counts = {}
        for d in difficulties:
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        
        difficulty_text = " | ".join([f"{k}: {v}" for k, v in difficulty_counts.items()])
        y = draw_wrapped(f"Difficulty Distribution: {difficulty_text}", y, leading=14)
        y -= 12
    
    for i, q in enumerate(quiz, 1):
        # Add difficulty indicator if available
        difficulty_text = f" [{q.get('difficulty', '')}]" if q.get('difficulty') and q.get('difficulty') != 'Not Specified' else ""
        y = draw_wrapped(f"{i}. {q['question']}{difficulty_text}", y, leading=14)
        for label, opt in zip(["A", "B", "C", "D"], q["options"]):
            y = draw_wrapped(f"   {label}) {opt}", y, leading=14)
        y -= 8
        if y < margin + 2*cm:
            c.showPage()
            y = height - margin
    
    c.showPage()
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Answer Key")
    y -= 20
    c.setFont("Helvetica", 11)
    for i, q in enumerate(quiz, 1):
        difficulty_text = f" [{q.get('difficulty', '')}]" if q.get('difficulty') and q.get('difficulty') != 'Not Specified' else ""
        y = draw_wrapped(f"{i}. {q['answer']}{difficulty_text}", y, leading=14)
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------------
# Streamlit App (Modified)
# ------------------------------
st.set_page_config(page_title="RAG Quiz Generator", page_icon="sparkles.png", layout="wide")
st.title("Enhanced RAG Quiz Generator with Difficulty Levels")

with st.sidebar:
    st.header("Upload & Settings")
    pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # Course selection dropdown
    available_courses = ["None"] + list(SAMPLE_QUESTIONS.keys())
    selected_course = st.selectbox(
        "Select Course (for difficulty-based questions)",
        available_courses,
        help="Select 'None' for general quiz generation without difficulty levels"
    )

    # Add the warning message when a course is selected
    if selected_course != "None":
        st.caption("📝 Make sure you've uploaded the same course PDF")
    
    num_q = st.number_input("Number of questions", min_value=1, max_value=50, value=10, step=1)
    top_k = st.slider("Retriever top-k passages per question", min_value=1, max_value=8, value=2)
    similarity_threshold = st.slider("Similarity threshold for deduplication", 
                                   min_value=0.5, max_value=0.95, value=0.85, step=0.05)
    
    # Show difficulty distribution if course is selected
    if selected_course != "None":
        easy_count = num_q // 3
        medium_count = num_q // 3
        hard_count = num_q - easy_count - medium_count
        st.info(f"**Difficulty Distribution:**\n- Easy: {easy_count}\n- Medium: {medium_count}\n- Hard: {hard_count}")
    
    st.caption("Stack: PyMuPDF + MiniLM + FAISS + Gemini Pro + ReportLab")

    if st.button("Build Quiz"):
        if not pdf_file:
            st.error("Please upload a PDF first.")
        else:
            # Ensure storage directory exists
            storage_dir = ensure_storage_dir()
            
            # Generate unique hash for the PDF
            pdf_hash = get_pdf_hash(pdf_file)
            
            # Check if we have cached version
            with st.spinner("Checking for cached index..."):
                cached_index, cached_chunks = load_faiss_from_disk(pdf_hash, storage_dir)
            
            if cached_index and cached_chunks:
                # Use cached version
                st.success("✅ Loaded from cache! Generating questions...")
                index = cached_index
                chunks = cached_chunks
                # For cached files, create readable text files too
                with st.spinner("Creating readable text files..."):
                    embedder = load_embedding_model()
                    embeddings = embedder.encode(chunks, convert_to_numpy=True)
                    create_readable_chunks_file(pdf_hash, chunks, storage_dir)
                    create_readable_vectors_file(pdf_hash, embeddings, storage_dir)
            else:
                # Build new index
                with st.spinner("Extracting PDF text..."):
                    text = extract_text_from_pdf(pdf_file)
                
                if len(text.strip()) < 50:
                    st.error("Could not extract enough text from the PDF.")
                else:
                    with st.spinner("Chunking + embedding + indexing..."):
                        chunks = chunk_text(text)
                        embedder = load_embedding_model()
                        # Build and save to disk with pdf_hash (this automatically creates text files)
                        index, embeddings = build_faiss_index(chunks, embedder, pdf_hash)
            
            # Continue with quiz generation
            if 'index' in locals() and 'chunks' in locals():
                with st.spinner("Generating questions with Gemini Pro..."):
                    # Pass selected_course to generate_diverse_quiz
                    course_for_generation = selected_course if selected_course != "None" else None
                    quiz = generate_diverse_quiz(int(num_q), chunks, index, embedder, top_k, course_for_generation)
                    
                    st.session_state["quiz"] = quiz
                    if quiz:
                        st.success(f"✅ Generated {len(quiz)} unique questions!")
                        
                        # Show cache status
                        if cached_index:
                            st.info("📁 Used cached FAISS index (fast!)")
                        else:
                            st.info("🔄 Built new FAISS index (cached for next time)")
                        
                        # Show question types and difficulty distribution
                        types_count = {}
                        difficulty_count = {}
                        for q in quiz:
                            q_type = q.get('query_type', 'unknown')
                            q_difficulty = q.get('difficulty', 'Not Specified')
                            types_count[q_type] = types_count.get(q_type, 0) + 1
                            difficulty_count[q_difficulty] = difficulty_count.get(q_difficulty, 0) + 1
                        
                        st.write("**Question Distribution:**")
                        if selected_course != "None":
                            st.write("**Difficulty Levels:**")
                            for difficulty, count in difficulty_count.items():
                                st.write(f"- {difficulty}: {count} questions")
                        
                    else:
                        st.error("Failed to generate any valid questions. Try with different settings.")
    
    st.markdown("---")
    st.subheader("Cache Management")
    
    if st.button("Clear All Cached Indexes"):
        storage_dir = ensure_storage_dir()
        faiss_files = [f for f in os.listdir(storage_dir) if f.endswith('.faiss')]
        for file in faiss_files:
            os.remove(f"{storage_dir}/{file}")
            # Also remove corresponding chunks file
            chunks_file = file.replace('.faiss', '_chunks.pkl')
            if os.path.exists(f"{storage_dir}/{chunks_file}"):
                os.remove(f"{storage_dir}/{chunks_file}")
            # Also remove corresponding text files
            txt_chunks_file = file.replace('.faiss', '_chunks.txt')
            if os.path.exists(f"{storage_dir}/{txt_chunks_file}"):
                os.remove(f"{storage_dir}/{txt_chunks_file}")
            txt_vectors_file = file.replace('.faiss', '_vectors.txt')
            if os.path.exists(f"{storage_dir}/{txt_vectors_file}"):
                os.remove(f"{storage_dir}/{txt_vectors_file}")
        st.success(f"Cleared {len(faiss_files)} cached indexes!")
    
    # Show cache info
    storage_dir = ensure_storage_dir()
    cached_files = [f for f in os.listdir(storage_dir) if f.endswith('.faiss')]
    st.caption(f"📁 Cached PDFs: {len(cached_files)}")

# Show quiz if available
quiz = st.session_state.get("quiz")
if quiz:
    st.subheader("📝 Attempt Quiz")
    
    # Show difficulty summary if available
    difficulties = [q.get('difficulty', 'Not Specified') for q in quiz]
    if any(d != 'Not Specified' for d in difficulties):
        difficulty_counts = {}
        for d in difficulties:
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        
        st.info(f"**Quiz Summary:** {', '.join([f'{k}: {v}' for k, v in difficulty_counts.items()])}")
    
    answers = {}
    with st.form("quiz_form"):
        for i, q in enumerate(quiz, 1):
            # Show difficulty indicator if available
            difficulty_text = f" [{q.get('difficulty', '')}]" if q.get('difficulty') and q.get('difficulty') != 'Not Specified' else ""
            st.markdown(f"**{i}. {q['question']}{difficulty_text}**")
            options_list = ["A", "B", "C", "D"]
            # Ensure there are 4 options before displaying
            display_options = q["options"] + [f"Option {chr(65+i)}" for i in range(4 - len(q["options"]))]
            
            choice = st.radio(
                f"Select your answer for Q{i}",
                options=options_list,
                format_func=lambda x: f"{x}) {display_options[options_list.index(x)]}",
                key=f"q{i}",
                horizontal=True
            )
            answers[i] = choice
            st.markdown("---")
        submitted = st.form_submit_button("Submit")
    if submitted:
        score = 0
        detailed = []
        for i, q in enumerate(quiz, 1):
            correct = q["answer"]
            user_ans = answers[i]
            is_right = (user_ans == correct)
            score += int(is_right)
            detailed.append((i, is_right, correct, user_ans, q.get('difficulty', 'Not Specified')))
        st.success(f"Your Score: {score} / {len(quiz)}")
        
        # Show score by difficulty if available
        if any(d[4] != 'Not Specified' for d in detailed):
            difficulty_scores = {}
            difficulty_total = {}
            for i, ok, corr, ua, diff in detailed:
                difficulty_scores[diff] = difficulty_scores.get(diff, 0) + int(ok)
                difficulty_total[diff] = difficulty_total.get(diff, 0) + 1
            
            st.write("**Score by Difficulty:**")
            for diff in difficulty_scores:
                st.write(f"- {diff}: {difficulty_scores[diff]}/{difficulty_total[diff]}")
        
        with st.expander("See detailed feedback"):
            for i, ok, corr, ua, diff in detailed:
                diff_text = f" [{diff}]" if diff != 'Not Specified' else ""
                st.write(f"Q{i}{diff_text}: {'✅ Correct' if ok else '❌ Incorrect'} | Your: {ua} | Correct: {corr}")

        buf = make_quiz_pdf(quiz, title="Generated MCQ Quiz")
        st.download_button(
            "⬇️ Download Quiz as PDF (with Answer Key)",
            data=buf,
            file_name="Quiz_With_Answers.pdf",
            mime="application/pdf"
        )
else:
    st.info("Upload a PDF and click **Build Quiz** from the sidebar to get started.")