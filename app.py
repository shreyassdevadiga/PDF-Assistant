#!/usr/bin/env python3
"""
app.py
Single-file foundation for "Voice Automated PDF Assistant" — Feature 1 (PDF loading & text extraction).

Capabilities:
- Load PDF safely
- Extract text per page (PyPDF2, optional pdfplumber fallback)
- Detect scanned/empty pages
- Export text (single file or per-page files)
- CLI interface + optional Streamlit preview

Usage (CLI):
    python app.py info path/to/file.pdf
    python app.py page path/to/file.pdf 0
    python app.py extract path/to/file.pdf --out all_text.txt
    python app.py extract path/to/file.pdf --out-dir pages_out

Streamlit:
    streamlit run app.py
"""

from __future__ import annotations
import sys
import os
import argparse
import logging
from typing import List, Tuple, Optional

# Try import robust libraries; handle gracefully if missing
try:
    from PyPDF2 import PdfReader
except Exception as e:
    print("ERROR: PyPDF2 is required. Install with `pip install PyPDF2`.")
    raise e

# Optional: pdfplumber gives better extraction for complex layouts
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

# Optional: OCR fallback (requires system Tesseract + pytesseract + pdf2image)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# Optional Streamlit UI
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pdf_assistant")

class PDFHandlerError(Exception):
    pass

class PDFHandler:
    """
    Simple PDF handler that wraps extraction logic.
    Prefer pdfplumber if available; otherwise use PyPDF2.
    """

    def __init__(self, path: str, use_plumber_if_available: bool = True):
        self.path = path
        self.reader = None
        self._num_pages = None
        self.use_plumber = use_plumber_if_available and _HAS_PDFPLUMBER
        if not os.path.isfile(path):
            raise PDFHandlerError(f"File not found: {path}")
        if not path.lower().endswith(".pdf"):
            raise PDFHandlerError("File is not a PDF (must have .pdf extension).")
        # Try basic open to raise helpful errors early
        try:
            self.reader = PdfReader(path)
            # access metadata to force parsing
            _ = self.reader.metadata
            self._num_pages = len(self.reader.pages)
        except Exception as e:
            raise PDFHandlerError(f"Failed to open PDF: {e}") from e

    def num_pages(self) -> int:
        return self._num_pages or 0

    def extract_text_by_page(self, page_no: int, attempt_plumber: Optional[bool]=None) -> str:
        """
        Extract text from a single page (0-indexed).
        attempt_plumber overrides default use of pdfplumber for this call.
        Returns empty string if nothing extracted (likely scanned page).
        """
        if page_no < 0 or page_no >= self.num_pages():
            raise IndexError("page_no out of range")
        # If pdfplumber is available and chosen, prefer it (often better)
        use_plumber = self.use_plumber if attempt_plumber is None else bool(attempt_plumber)
        if use_plumber and _HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(self.path) as pdf:
                    page = pdf.pages[page_no]
                    text = page.extract_text() or ""
                    return text.strip()
            except Exception as e:
                logger.debug("pdfplumber extraction failed for page %d: %s", page_no, e)
                # fallback to PyPDF2
        # PyPDF2 fallback
        try:
            page = self.reader.pages[page_no]
            text = page.extract_text() or ""
            return text.strip()
        except Exception as e:
            logger.debug("PyPDF2 extraction failed for page %d: %s", page_no, e)
            return ""

    def extract_all_pages(self, attempt_plumber: Optional[bool]=None) -> List[str]:
        """Extract text from all pages and return list[str] of length num_pages"""
        texts = []
        for i in range(self.num_pages()):
            texts.append(self.extract_text_by_page(i, attempt_plumber=attempt_plumber))
        return texts

    def pages_with_no_text(self, attempt_plumber: Optional[bool]=None) -> List[int]:
        """Return list of 0-indexed page numbers that likely are scanned / have no text"""
        texts = self.extract_all_pages(attempt_plumber=attempt_plumber)
        empty = [i for i,t in enumerate(texts) if not t or len(t.strip())==0]
        return empty

    # OCR helper (optional): converts page images to text using pytesseract
    def ocr_page(self, page_no: int, dpi: int = 200, lang: str = "eng") -> str:
        if not _HAS_OCR:
            raise PDFHandlerError("OCR not available: install pytesseract + pdf2image + pillow and system Tesseract")
        if page_no < 0 or page_no >= self.num_pages():
            raise IndexError("page_no out of range")
        try:
            # convert single page (page numbers for convert_from_path are 1-indexed)
            images = convert_from_path(self.path, dpi=dpi, first_page=page_no+1, last_page=page_no+1)
            if not images:
                return ""
            img: Image.Image = images[0]
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        except Exception as e:
            raise PDFHandlerError(f"OCR failed for page {page_no}: {e}") from e

    def extract_all_with_ocr_on_empty(self, dpi: int = 200, lang: str = "eng") -> List[str]:
        """
        Extract text for all pages; for empty pages, attempt OCR if available.
        This can be slower due to image conversion.
        """
        texts = self.extract_all_pages()
        if _HAS_OCR:
            for i, t in enumerate(texts):
                if not t or len(t.strip())==0:
                    try:
                        ocr_t = self.ocr_page(i, dpi=dpi, lang=lang)
                        if ocr_t:
                            texts[i] = ocr_t
                    except Exception as e:
                        logger.warning("OCR for page %d failed: %s", i, e)
        return texts

    # Export helpers
    def save_all_text_single_file(self, out_path: str, join_with="\n\n---PAGE-BREAK---\n\n") -> None:
        texts = self.extract_all_pages()
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(join_with.join(texts))
        except Exception as e:
            raise PDFHandlerError(f"Failed to write to {out_path}: {e}") from e

    def save_pages_to_dir(self, out_dir: str, prefix: str = "page") -> None:
        os.makedirs(out_dir, exist_ok=True)
        texts = self.extract_all_pages()
        for i, t in enumerate(texts):
            filename = os.path.join(out_dir, f"{prefix}_{i+1:03d}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(t)

# -------------------------
# CLI
# -------------------------
def cli_main(argv=None):
    parser = argparse.ArgumentParser(prog="app.py", description="PDF extraction utilities (Feature 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Show PDF info (pages, empty pages)")
    p_info.add_argument("pdf", help="Path to PDF")

    p_page = sub.add_parser("page", help="Print a page's extracted text")
    p_page.add_argument("pdf", help="Path to PDF")
    p_page.add_argument("page_no", type=int, help="0-indexed page number")
    p_page.add_argument("--try-plumber", action="store_true", help="Try pdfplumber extraction even if not auto-chosen")

    p_extract = sub.add_parser("extract", help="Extract text from PDF")
    p_extract.add_argument("pdf", help="Path to PDF")
    p_extract.add_argument("--out", "-o", help="Save combined text to file (single file)")
    p_extract.add_argument("--out-dir", help="Save per-page text files into directory")
    p_extract.add_argument("--ocr-if-empty", action="store_true", help="If pages are empty, attempt OCR (slow)")

    args = parser.parse_args(argv)

    try:
        handler = PDFHandler(args.pdf)
    except PDFHandlerError as e:
        logger.error(e)
        sys.exit(2)

    if args.cmd == "info":
        n = handler.num_pages()
        empty = handler.pages_with_no_text()
        print(f"File: {args.pdf}")
        print(f"Pages: {n}")
        if empty:
            print(f"Pages with no extracted text (likely scanned): {empty}")
            if _HAS_OCR:
                print("OCR is available in this environment; use 'extract --ocr-if-empty' to attempt OCR on empty pages.")
            else:
                print("OCR not available. To enable OCR install pytesseract, pdf2image, pillow and system Tesseract.")
        else:
            print("All pages have some extracted text.")
        return

    if args.cmd == "page":
        try:
            text = handler.extract_text_by_page(args.page_no, attempt_plumber=args.try_plumber)
            if not text:
                print("[No text extracted from this page]")
            else:
                print(text)
        except Exception as e:
            logger.error("Failed to extract page: %s", e)
            sys.exit(3)
        return

    if args.cmd == "extract":
        if args.out:
            if args.ocr_if_empty:
                texts = handler.extract_all_with_ocr_on_empty()
                try:
                    with open(args.out, "w", encoding="utf-8") as f:
                        f.write("\n\n---PAGE-BREAK---\n\n".join(texts))
                    print(f"Wrote combined text (OCR attempted on empty pages) to {args.out}")
                except Exception as e:
                    logger.error("Failed to write file: %s", e)
                    sys.exit(4)
            else:
                handler.save_all_text_single_file(args.out)
                print(f"Wrote combined text to {args.out}")
        if args.out_dir:
            handler.save_pages_to_dir(args.out_dir)
            print(f"Wrote per-page text files to {args.out_dir}")
        if not args.out and not args.out_dir:
            # default: print a short summary to stdout
            texts = handler.extract_all_pages()
            for i, t in enumerate(texts):
                print(f"--- PAGE {i+1} ---")
                if not t:
                    print("[No text extracted from this page]")
                else:
                    # show first 1000 chars
                    print(t[:1000])
                    if len(t) > 1000:
                        print("... (truncated)")

# -------------------------
# Simple Streamlit preview (optional)
# -------------------------
def streamlit_app():
    st.title("PDF Extraction — Feature 1 (Robust PDF Loading & Extraction)")
    st.write("Upload a PDF and test per-page extraction. This preview is minimal — it's for quick testing.")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF to begin.")
        return
    # save temporarily
    tmp_path = os.path.join(".", "_tmp_uploaded.pdf")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        handler = PDFHandler(tmp_path)
    except PDFHandlerError as e:
        st.error(f"Failed to open PDF: {e}")
        return

    st.markdown(f"**Loaded:** `{uploaded.name}` — pages: **{handler.num_pages()}**")
    empty_pages = handler.pages_with_no_text()
    if empty_pages:
        st.warning(f"Pages with no extracted text (likely scanned): {empty_pages}")
    page_sel = st.number_input("Page (0-indexed)", min_value=0, max_value=max(0, handler.num_pages()-1), value=0)
    use_plumber = st.checkbox("Prefer pdfplumber extraction (if available)", value=True)
    if st.button("Extract page"):
        txt = handler.extract_text_by_page(int(page_sel), attempt_plumber=use_plumber)
        if not txt:
            st.info("No text extracted from this page.")
        else:
            st.text_area("Extracted text", value=txt, height=300)
    if st.button("Extract all and show summary"):
        texts = handler.extract_all_pages()
        for i,t in enumerate(texts):
            st.markdown(f"**Page {i+1}** — {'(empty)' if not t else f'{len(t)} chars'}")
            if t:
                st.write(t[:500] + ("..." if len(t) > 500 else ""))

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    # If running under streamlit, streamlit executes the file but sets special env var
    if _HAS_STREAMLIT and "streamlit" in sys.argv[0]:
        # streamlit runs this file and expects to execute a script; avoid CLI branching
        streamlit_app()
    # detect if user invoked 'streamlit run app.py' (streamlit will import the file and call it)
    # For direct python invocation, use CLI
    if len(sys.argv) > 1 and _HAS_STREAMLIT and sys.argv[1] == "streamlit":
        # fallback
        streamlit_app()
    elif _HAS_STREAMLIT and any(arg.startswith("--server.port") for arg in sys.argv):
        # heuristic: streamlit may add args — still run streamlit app
        streamlit_app()
    else:
        # If no additional CLI arguments are provided, avoid argparse error.
        # Prefer launching the Streamlit preview when available; otherwise show usage.
        if len(sys.argv) == 1:
            if _HAS_STREAMLIT:
                streamlit_app()
            else:
                print("Usage (CLI):")
                print("  python app.py info path/to/file.pdf")
                print("  python app.py page path/to/file.pdf 0")
                print("  python app.py extract path/to/file.pdf --out all_text.txt")
                print("  python app.py extract path/to/file.pdf --out-dir pages_out")
        else:
            cli_main()
