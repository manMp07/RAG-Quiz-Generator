# pages/4_Previous_Questions.py
import streamlit as st
from utils.db import get_questions_by_user
from utils.auth import logout
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import io
import textwrap

# Authentication check
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to view your questions.")
    st.page_link("pages/1_Login.py", label="Go to Login")
    st.stop()

st.set_page_config(page_title="Question Bank", page_icon="📚")
st.title("📚 Previously Generated Questions")
st.caption(f"Logged in as: {st.session_state['user_email']}")

# Logout in sidebar
with st.sidebar:
    if st.button("Logout"):
        logout()
        st.switch_page("pages/1_Login.py")

# Fetch all questions for this user
user_id = st.session_state["user_id"]
all_questions = get_questions_by_user(user_id)

if not all_questions:
    st.info("No questions saved yet. Generate and save a quiz first!")
    st.stop()

# Group by course
questions_by_course = {}
for q in all_questions:
    course = q.get("course", "Uncategorized")
    if course not in questions_by_course:
        questions_by_course[course] = []
    questions_by_course[course].append(q)

# For each course, show questions with checkboxes
selected_questions = []  # list to collect selected question dicts

for course, qlist in questions_by_course.items():
    st.subheader(f"📖 {course}")
    with st.expander(f"Show {len(qlist)} questions", expanded=False):
        for q in qlist:
            cols = st.columns([0.05, 0.7, 0.25])
            with cols[0]:
                # Checkbox for selection
                if st.checkbox("", key=f"sel_{q['_id']}"):
                    selected_questions.append(q)
            with cols[1]:
                st.markdown(f"**{q['question_text']}**")
                st.markdown(f"A) {q['options'][0]}  |  B) {q['options'][1]}  |  C) {q['options'][2]}  |  D) {q['options'][3]}")
                st.markdown(f"*Correct: {q['correct_answer']}*")
            with cols[2]:
                st.markdown(f"**Difficulty:** {q['difficulty']}")
            st.divider()

# PDF generation function (copied from generator, but can be placed here)
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

    # Answer Key
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

# Create Quiz button
if selected_questions:
    st.success(f"Selected {len(selected_questions)} questions.")
    if st.button("📄 Create Quiz PDF from Selected"):
        # Generate PDF
        pdf_buffer = make_selected_questions_pdf(selected_questions, title=f"Custom Quiz - {course}")
        st.download_button(
            label="⬇️ Download Quiz PDF",
            data=pdf_buffer,
            file_name="custom_quiz.pdf",
            mime="application/pdf"
        )
else:
    st.info("Select questions using the checkboxes above to create a quiz.")
