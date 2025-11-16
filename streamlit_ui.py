import os

import streamlit as st

from app import PDFHandler, PDFHandlerError


st.set_page_config(page_title="PDF Assistant - Viewer", layout="wide")
st.title("PDF Assistant – PDF Viewer & Extractor")

st.write(
    "Upload a PDF, view basic info, and inspect text from individual pages. "
    "Voice commands and advanced CLI features are available via `python app.py` (see help there)."
)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

# Save uploaded file to a temporary path in the working directory
TMP_DIR = ".streamlit_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

pdf_path = os.path.join(TMP_DIR, uploaded.name)
with open(pdf_path, "wb") as f:
    f.write(uploaded.getbuffer())

try:
    handler = PDFHandler(pdf_path)
except PDFHandlerError as e:
    st.error(f"Failed to open PDF: {e}")
    st.stop()

st.success(f"Loaded `{uploaded.name}` with **{handler.num_pages()}** pages.")

col_info, col_page = st.columns([1, 2])

with col_info:
    st.subheader("Document Info")
    st.write(f"**File path:** `{pdf_path}`")
    st.write(f"**Pages:** {handler.num_pages()}")

with col_page:
    st.subheader("Page Viewer")
    page_no = st.number_input(
        "Page (0-indexed)",
        min_value=0,
        max_value=max(0, handler.num_pages() - 1),
        value=0,
        step=1,
    )
    if st.button("Extract page text"):
        try:
            text = handler.extract_text_by_page(int(page_no))
            if not text:
                st.info("No text extracted from this page (it may be scanned-only).")
            else:
                st.text_area("Extracted text", value=text, height=400)
        except Exception as e:
            st.error(f"Failed to extract page: {e}")
