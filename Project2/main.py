"""
Resume Extractor
----------------
Upload a resume file (PDF, DOCX, or TXT) → get structured JSON using OpenAI + Pydantic.
If parsing fails, the validation error is fed back into the prompt and retried.
"""

import io
import json
import os
from typing import List, Optional

import pdfplumber
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, EmailStr, ValidationError

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Pydantic schema ───────────────────────────────────────────────────────────
class ResumeData(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = []
    experience: List[str] = []   # e.g. "Software Engineer at Google (2020-2023)"
    education: List[str] = []    # e.g. "B.Tech CS - MIT (2016-2020)"
    languages: List[str] = []
    certifications: List[str] = []

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a resume parser. Extract information from the resume text and return ONLY a valid JSON object.
Do NOT include markdown, code fences, or any explanation — just raw JSON.

The JSON must match this exact schema:
{
  "name": "string (required)",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "summary": "string or null",
  "skills": ["list of skills"],
  "experience": ["list of job titles and companies, e.g. 'Software Engineer at Google'"],
  "education": ["list of degrees and institutions, e.g. 'B.Tech in CS from MIT'"],
  "languages": ["list of languages"],
  "certifications": ["list of certifications"]
}
"""

# ── File text extractors ──────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract all text from a DOCX file."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text_from_file(uploaded_file) -> str:
    """
    Detect file type by extension and extract plain text.
    Supports: PDF, DOCX, TXT
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

# ── Core extraction function with retry logic ─────────────────────────────────
def extract_resume(resume_text: str, max_retries: int = 3) -> ResumeData:
    """
    Call OpenAI to extract resume fields.
    If JSON parsing or Pydantic validation fails, retry with the error fed back.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Parse this resume:\n\n{resume_text}"},
    ]

    for attempt in range(1, max_retries + 1):
        st.write(f"Attempt {attempt} / {max_retries} ...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        with st.expander(f"Raw LLM output (attempt {attempt})"):
            st.code(raw, language="json")

        # Try to parse and validate
        try:
            data = json.loads(raw)
            resume = ResumeData(**data)
            return resume                          # success - return early

        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = str(e)
            st.warning(f"Attempt {attempt} failed:\n```\n{error_msg}\n```")

            if attempt == max_retries:
                raise                              # give up after max retries

            # Feed the error back so the model can self-correct
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your response caused this error:\n{error_msg}\n\n"
                    "Please fix it and return ONLY valid JSON matching the schema."
                ),
            })

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Extractor", page_icon="📄", layout="wide")

st.title("📄 Resume Extractor")
st.caption("Upload a resume file → get structured JSON using OpenAI + Pydantic validation")

# File uploader — accepts PDF, DOCX, TXT
uploaded_file = st.file_uploader(
    label="Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT",
)

max_retries = st.slider("Max retries on validation failure", min_value=1, max_value=5, value=3)

if st.button("Extract Resume", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a resume file first.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set in your .env file.")
    else:
        with st.spinner("Reading file..."):
            try:
                resume_text = extract_text_from_file(uploaded_file)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

        # Show extracted text so user can verify it was read correctly
        with st.expander("Extracted text from file"):
            st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

        with st.spinner("Extracting structured data..."):
            try:
                result: ResumeData = extract_resume(resume_text, max_retries=max_retries)

                st.success("Extraction successful!")

                # Show structured results in two columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Personal Info")
                    st.write(f"**Name:** {result.name}")
                    st.write(f"**Email:** {result.email or 'N/A'}")
                    st.write(f"**Phone:** {result.phone or 'N/A'}")
                    st.write(f"**Location:** {result.location or 'N/A'}")
                    if result.summary:
                        st.write(f"**Summary:** {result.summary}")

                    st.subheader("Skills")
                    st.write(", ".join(result.skills) if result.skills else "N/A")

                    st.subheader("Languages")
                    st.write(", ".join(result.languages) if result.languages else "N/A")

                with col2:
                    st.subheader("Experience")
                    for exp in result.experience:
                        st.write(f"- {exp}")
                    if not result.experience:
                        st.write("N/A")

                    st.subheader("Education")
                    for edu in result.education:
                        st.write(f"- {edu}")
                    if not result.education:
                        st.write("N/A")

                    st.subheader("Certifications")
                    for cert in result.certifications:
                        st.write(f"- {cert}")
                    if not result.certifications:
                        st.write("N/A")

                # Raw validated JSON
                st.subheader("Validated JSON")
                st.json(result.model_dump())

            except Exception as e:
                st.error(f"Failed after all retries:\n\n{e}")
