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
    """Generate PDF with proper hanging indents, spacing, and full-question page breaks."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    left_x = margin
    right_limit = width - margin
    y = height - margin

    c.setFont("Helvetica", 11)
    line_height = 14
    question_text_indent = 20   # space for "1. "
    option_label_indent = 0
    option_text_indent = 24     # space for "A) "
    extra_spacing = 4
    bottom_margin = margin

    # Helper: draw wrapped text with hanging indent, return new y and number of lines
    def draw_wrapped_with_hanging(text, x_start, hanging_indent, y_pos, max_width, leading):
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if c.stringWidth(test_line, "Helvetica", 11) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for i, line in enumerate(lines):
            if y_pos < bottom_margin + leading:
                # Not enough space – caller should have prevented this
                c.showPage()
                c.setFont("Helvetica", 11)
                y_pos = height - margin
            if i == 0:
                c.drawString(x_start, y_pos, line)
            else:
                c.drawString(hanging_indent, y_pos, line)
            y_pos -= leading
        return y_pos, len(lines)

    # Calculate total height needed for a question (including spacing)
    def calculate_question_height(question, max_text_width, max_opt_width):
        # Question text lines
        text_part = question['question_text']
        diff = question.get('difficulty', 'Not Specified')
        if diff != 'Not Specified':
            text_part += f" [{diff}]"
        # Wrapping logic (same as draw)
        q_lines = []
        words = text_part.split()
        current = ""
        for w in words:
            test = current + (" " if current else "") + w
            if c.stringWidth(test, "Helvetica", 11) <= max_text_width:
                current = test
            else:
                if current:
                    q_lines.append(current)
                current = w
        if current:
            q_lines.append(current)
        height = len(q_lines) * line_height + extra_spacing  # space before options

        # Options
        for opt in question['options']:
            opt_lines = []
            words = opt.split()
            current = ""
            for w in words:
                test = current + (" " if current else "") + w
                if c.stringWidth(test, "Helvetica", 11) <= max_opt_width:
                    current = test
                else:
                    if current:
                        opt_lines.append(current)
                    current = w
            if current:
                opt_lines.append(current)
            height += len(opt_lines) * line_height + extra_spacing
        # Add one line of spacing after the question
        height += line_height
        return height

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_x, y, title)
    y -= 24
    c.setFont("Helvetica", 11)

    max_text_width = right_limit - (left_x + question_text_indent)
    max_opt_width = right_limit - (left_x + option_text_indent)

    for idx, q in enumerate(questions, 1):
        # Calculate required height for this question
        required = calculate_question_height(q, max_text_width, max_opt_width)
        # Check if there's enough space on current page
        if y - required < bottom_margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 11)

        # Draw the question
        number_part = f"{idx}."
        text_part = q['question_text']
        diff = q.get('difficulty', 'Not Specified')
        if diff != 'Not Specified':
            text_part += f" [{diff}]"

        # Question number
        c.drawString(left_x, y, number_part)
        # Question text
        y, _ = draw_wrapped_with_hanging(
            text_part,
            left_x + question_text_indent,
            left_x + question_text_indent,
            y,
            max_text_width,
            line_height
        )
        y -= extra_spacing

        # Options
        for label, opt in zip(["A", "B", "C", "D"], q["options"]):
            label_part = f"{label})"
            c.drawString(left_x + option_label_indent, y, label_part)
            y, _ = draw_wrapped_with_hanging(
                opt,
                left_x + option_text_indent,
                left_x + option_text_indent,
                y,
                max_opt_width,
                line_height
            )
            y -= extra_spacing

        # Space after question
        y -= line_height

    # Answer Key – new page
    c.showPage()
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_x, y, "Answer Key")
    y -= 20
    c.setFont("Helvetica", 11)
    for i, q in enumerate(questions, 1):
        ans_text = f"{i}. {q['correct_answer']}"
        y, _ = draw_wrapped_with_hanging(ans_text, left_x, left_x, y, right_limit - left_x, line_height)
        y -= 2

    c.save()
    buffer.seek(0)
    return buffer

# Create Quiz button
if selected_questions:
    st.success(f"Selected {len(selected_questions)} questions.")
    if st.button("📄 Create Quiz PDF from Selected"):
        # Generate PDF
        pdf_buffer = make_selected_questions_pdf(selected_questions, title=f"Custom Quiz - Mobile Computing")
        st.download_button(
            label="⬇️ Download Quiz PDF",
            data=pdf_buffer,
            file_name="custom_quiz.pdf",
            mime="application/pdf"
        )
else:
    st.info("Select questions using the checkboxes above to create a quiz.")
