#!/usr/bin/env python3
"""
Streamlit-only Voice PDF Assistant — direct browser microphone (no streamlit-webrtc)

Requirements:
    pip install streamlit PyPDF2 pdfplumber SpeechRecognition gTTS streamlit-audiorecorder

Run:
    streamlit run app.py

Features:
 - Upload PDF, extract per-page text (pdfplumber preferred if installed)
 - Navigate pages (Next / Previous / Go-to)
 - Read page (gTTS -> audio played in browser)
 - Live microphone recording directly in browser (start/stop) using streamlit-audiorecorder
   -> recorded WAV bytes are sent to Python and transcribed with SpeechRecognition
 - Recognized speech is parsed into commands (next, previous, go to page N, read, search, summarize)
"""

from __future__ import annotations
import io
import re
import logging
from typing import List, Dict, Optional

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("streamlit_voice_pdf_recorder")

# ---------- Helper: in-browser recorder component ----------
# This uses the lightweight streamlit-audiorecorder component.
# It returns either (None) when no recording, or a tuple: (audio_bytes, sample_rate)
# The component's API: `from streamlit_audiorecorder import audiorecorder`
# audiorecorder() returns bytes (wav) when recording stops.
try:
    from streamlit_audiorecorder import audiorecorder  # type: ignore
    _HAS_AUDIOREC = True
except Exception:
    _HAS_AUDIOREC = False

# ---------- PDF extraction (lazy) ----------
def extract_pages_from_bytes(pdf_bytes: bytes, use_pdfplumber: bool = True) -> List[str]:
    """
    Extract text from every page of the uploaded PDF.
    Uses pdfplumber when available (often better), else PyPDF2.
    """
    texts: List[str] = []
    if use_pdfplumber:
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    texts.append((p.extract_text() or "").strip())
            return texts
        except Exception as e:
            logger.debug("pdfplumber not available or failed: %s", e)

    # PyPDF2 fallback
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i in range(len(reader.pages)):
            try:
                texts.append((reader.pages[i].extract_text() or "").strip())
            except Exception:
                texts.append("")
        return texts
    except Exception as e:
        logger.exception("PDF extraction failed")
        raise RuntimeError(f"PDF extraction failed: {e}") from e

# ---------- Simple PDF handler for external UI (streamlit_ui.py) ----------
class PDFHandlerError(Exception):
    """Custom exception for PDFHandler-related errors."""


class PDFHandler:
    """Minimal PDF handler used by streamlit_ui.py.

    It wraps the existing extract_pages_from_bytes() helper to provide
    page counting and per-page text extraction from a PDF file path.
    """

    def __init__(self, pdf_path: str, use_pdfplumber: bool = True):
        self.pdf_path = pdf_path
        self._use_pdfplumber = use_pdfplumber
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception as e:
            raise PDFHandlerError(f"Failed to open PDF '{pdf_path}': {e}") from e

        try:
            self._pages = extract_pages_from_bytes(pdf_bytes, use_pdfplumber=self._use_pdfplumber)
        except Exception as e:
            raise PDFHandlerError(f"Failed to extract PDF pages: {e}") from e

        if not isinstance(self._pages, list):
            raise PDFHandlerError("PDF extraction did not return a list of page texts.")

    def num_pages(self) -> int:
        return len(self._pages)

    def extract_text_by_page(self, page_index: int) -> str:
        try:
            return self._pages[page_index]
        except IndexError as e:
            raise PDFHandlerError(f"Page index out of range: {page_index}") from e

# ---------- TTS ----------
def generate_tts_mp3_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    try:
        from gtts import gTTS  # type: ignore
    except Exception:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning("gTTS failed: %s", e)
        return None

# ---------- Command parser ----------
def parse_command(recognized_text: str) -> Dict:
    if not recognized_text:
        return {"action":"none"}
    txt = recognized_text.lower().strip()
    if re.search(r"\b(next|next page|go to next)\b", txt):
        return {"action":"next"}
    if re.search(r"\b(previous|prev|previous page|go back)\b", txt):
        return {"action":"previous"}
    if re.search(r"\b(stop listening|stop|exit|quit)\b", txt):
        return {"action":"stop"}
    if re.search(r"\b(read (this )?page|read page|read)\b", txt):
        m = re.search(r"page (\d+)", txt)
        if m:
            return {"action":"read", "page": int(m.group(1))-1}
        return {"action":"read"}
    m = re.search(r"go to page (\d+)", txt)
    if m:
        return {"action":"goto", "page": int(m.group(1))-1}
    m2 = re.search(r"\b(go to|goto|open|page)\s+(\d+)\b", txt)
    if m2:
        return {"action":"goto", "page": int(m2.group(2))-1}
    m = re.search(r"(search for|find|look for|search)\s+(.*)", txt)
    if m:
        return {"action":"search", "query": m.group(2).strip()}
    if re.search(r"\b(summarize|summary|give me a summary|summarize this)\b", txt):
        return {"action":"summarize"}
    m = re.search(r"(highlight|mark)\s+(.*)", txt)
    if m:
        return {"action":"highlight", "text": m.group(2).strip()}
    m = re.search(r"(annotate|add note|note)\s+(.*)", txt)
    if m:
        return {"action":"annotate", "text": m.group(2).strip()}
    m = re.search(r"page\s+(\d+)", txt)
    if m:
        return {"action":"goto", "page": int(m.group(1))-1}
    return {"action":"unknown", "text": recognized_text}

# ---------- Recognize WAV bytes (SpeechRecognition) ----------
def recognize_wav_bytes(wav_bytes: bytes) -> Optional[str]:
    """
    Use SpeechRecognition to transcribe WAV audio bytes using Google Web Speech API.
    Requires `pip install SpeechRecognition`.
    """
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        raise RuntimeError("SpeechRecognition is not installed. Install with `pip install SpeechRecognition`.")

    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = r.record(src)
            text = r.recognize_google(audio)  # online Google Web Speech API
            return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition request failed (probably no internet): {e}")
    except Exception as e:
        raise RuntimeError(f"Audio recognition failed: {e}")

# ---------- Streamlit app ----------
def streamlit_app():
    st.set_page_config(page_title="Voice PDF Assistant (Browser Mic)", layout="wide")
    st.title("Voice Automated PDF Assistant — Browser Microphone Commands")

    st.markdown("""
    - Upload a PDF.
    - Use **Start recording** / **Stop** (in the recorder widget) to capture a short voice command.
    - The app transcribes the command and executes it: *next*, *previous*, *go to page N*, *read page*, *search for ...*.
    """)

    st.sidebar.header("Settings")
    use_pdfplumber = st.sidebar.checkbox("Prefer pdfplumber extraction (if available)", value=True)
    tts_lang = st.sidebar.text_input("TTS language (gTTS)", value="en")
    snippet_len = st.sidebar.number_input("Search snippet length", min_value=50, max_value=1000, value=300)

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is None:
        st.info("Upload a PDF to begin.")
        return

    pdf_bytes = uploaded_pdf.read()

    @st.cache_data(show_spinner=False)
    def cached_extract_all(b: bytes, use_pdfpl: bool):
        return extract_pages_from_bytes(b, use_pdfplumber=use_pdfpl)

    try:
        pages = cached_extract_all(pdf_bytes, use_pdfplumber)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return

    n_pages = len(pages)
    st.success(f"Loaded {uploaded_pdf.name} — {n_pages} pages")

    # session state
    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0

    col1, col2, col3, col4 = st.columns([1,1,1,2])
    with col1:
        if st.button("Previous"):
            if st.session_state.page_idx > 0:
                st.session_state.page_idx -= 1
    with col2:
        if st.button("Next"):
            if st.session_state.page_idx + 1 < n_pages:
                st.session_state.page_idx += 1
    with col3:
        goto = st.number_input("Go to page (1-indexed)", min_value=1, max_value=max(1,n_pages), value=st.session_state.page_idx+1)
        if st.button("Go"):
            st.session_state.page_idx = int(goto)-1
    with col4:
        if st.button("Read page (gTTS)"):
            text = pages[st.session_state.page_idx]
            if not text:
                st.info("No text extracted on this page to read.")
            else:
                mp3 = generate_tts_mp3_bytes(text[:15000], lang=tts_lang)
                if mp3:
                    st.audio(mp3, format="audio/mp3")
                else:
                    st.text_area("Page text", value=text, height=400)

    st.markdown("---")
    st.subheader(f"Page {st.session_state.page_idx+1}/{n_pages}")
    page_text = pages[st.session_state.page_idx]
    if not page_text:
        st.info("This page appears to have no extracted text (likely scanned).")
    st.text_area("Extracted page text", value=page_text, height=350)

    st.markdown("---")
    st.header("Record voice command (browser mic)")
    if not _HAS_AUDIOREC:
        st.warning("`streamlit-audiorecorder` component is not installed. Install with:\n\npip install streamlit-audiorecorder\n\nAfter install, reload this page.")
        st.info("Fallback: you can still upload an audio file for commands.")
    else:
        st.write("Press **Start / Stop** below to record a short voice command (1-8 seconds).")
        # audiorecorder returns bytes (wav) or None
        wav_bytes = audiorecorder(duration=8,  # max record duration seconds, adjust as needed
                                  key="recorder")  # returns wav bytes OR None
        if wav_bytes:
            st.success("Recording captured. Transcribing...")
            try:
                recognized = recognize_wav_bytes(wav_bytes)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                recognized = None
            if not recognized:
                st.warning("Could not transcribe audio.")
            else:
                st.success(f"Recognized: {recognized}")
                cmd = parse_command(recognized)
                st.write("Parsed command:", cmd)
                action = cmd.get("action")
                if action == "next":
                    if st.session_state.page_idx + 1 < n_pages:
                        st.session_state.page_idx += 1
                        st.info(f"Moved to page {st.session_state.page_idx+1}")
                    else:
                        st.info("Already at last page.")
                elif action == "previous":
                    if st.session_state.page_idx - 1 >= 0:
                        st.session_state.page_idx -= 1
                        st.info(f"Moved to page {st.session_state.page_idx+1}")
                    else:
                        st.info("Already at first page.")
                elif action in ("goto", "read"):
                    page = cmd.get("page")
                    if page is None:
                        page = st.session_state.page_idx
                    if page < 0 or page >= n_pages:
                        st.error("Requested page out of range.")
                    else:
                        st.session_state.page_idx = page
                        if action == "read":
                            txt = pages[page]
                            if not txt:
                                st.info("No text on this page to read.")
                            else:
                                mp3 = generate_tts_mp3_bytes(txt[:15000], lang=tts_lang)
                                if mp3:
                                    st.audio(mp3, format="audio/mp3")
                                else:
                                    st.text_area("Page text", value=txt, height=400)
                        else:
                            st.info(f"Moved to page {page+1}")
                elif action == "search":
                    q = cmd.get("query","")
                    if not q:
                        st.info("No search query detected.")
                    else:
                        sq = q.lower().strip()
                        found = []
                        for i, txt in enumerate(pages):
                            if txt and sq in txt.lower():
                                idx = txt.lower().find(sq)
                                start = max(0, idx-60)
                                snippet = txt[start:start+snippet_len].replace("\n"," ")
                                found.append((i+1, snippet))
                        if found:
                            st.success(f"Found on {len(found)} page(s): {[p for p,_ in found]}")
                            for p, snip in found[:50]:
                                st.markdown(f"**Page {p}** — ...{snip}...")
                        else:
                            st.info("No results found.")
                elif action == "summarize":
                    st.info("Summarization requested — not implemented yet.")
                else:
                    st.info("Unknown or unimplemented command.")
        else:
            st.info("No recording yet. Press Start, speak, then Stop.")

    st.markdown("---")
    st.header("Fallback: upload audio file (WAV/MP3) for commands")
    audio_file = st.file_uploader("Upload audio command file", type=["wav","mp3","ogg","m4a","flac","aiff"])
    if audio_file is not None:
        st.info("Processing uploaded audio.")
        audio_bytes = audio_file.read()
        fname = audio_file.name.lower()
        wav_bytes2 = None
        # if mp3/etc convert to wav using pydub if installed (optional)
        if fname.endswith(".mp3") or fname.endswith(".ogg") or fname.endswith(".m4a"):
            try:
                from pydub import AudioSegment  # type: ignore
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                out = io.BytesIO()
                audio.export(out, format="wav")
                out.seek(0)
                wav_bytes2 = out.read()
            except Exception as e:
                st.error("MP3/OGG conversion failed. Install pydub+ffmpeg or upload WAV file.")
                wav_bytes2 = None
        else:
            wav_bytes2 = audio_bytes
        if wav_bytes2:
            try:
                recognized = recognize_wav_bytes(wav_bytes2)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                recognized = None
            if not recognized:
                st.warning("Could not transcribe uploaded audio.")
            else:
                st.success(f"Recognized: {recognized}")
                cmd = parse_command(recognized)
                st.write("Parsed command:", cmd)
                # reuse same execution logic as above (for brevity not repeated)
                # Execute simple navigation/read/search as above
                action = cmd.get("action")
                if action == "next":
                    if st.session_state.page_idx + 1 < n_pages:
                        st.session_state.page_idx += 1
                        st.info(f"Moved to page {st.session_state.page_idx+1}")
                    else:
                        st.info("Already at last page.")
                elif action == "previous":
                    if st.session_state.page_idx - 1 >= 0:
                        st.session_state.page_idx -= 1
                        st.info(f"Moved to page {st.session_state.page_idx+1}")
                    else:
                        st.info("Already at first page.")
                elif action in ("goto", "read"):
                    page = cmd.get("page")
                    if page is None:
                        page = st.session_state.page_idx
                    if page < 0 or page >= n_pages:
                        st.error("Requested page out of range.")
                    else:
                        st.session_state.page_idx = page
                        if action == "read":
                            txt = pages[page]
                            if not txt:
                                st.info("No text to read.")
                            else:
                                mp3 = generate_tts_mp3_bytes(txt[:15000], lang=tts_lang)
                                if mp3:
                                    st.audio(mp3, format="audio/mp3")
                                else:
                                    st.text_area("Page text", value=txt, height=400)
                        else:
                            st.info(f"Moved to page {page+1}")
                elif action == "search":
                    q = cmd.get("query","")
                    if not q:
                        st.info("No search query detected.")
                    else:
                        sq = q.lower().strip()
                        found = []
                        for i, txt in enumerate(pages):
                            if txt and sq in txt.lower():
                                idx = txt.lower().find(sq)
                                start = max(0, idx-60)
                                snippet = txt[start:start+snippet_len].replace("\n"," ")
                                found.append((i+1, snippet))
                        if found:
                            st.success(f"Found on {len(found)} page(s): {[p for p,_ in found]}")
                            for p, snip in found[:50]:
                                st.markdown(f"**Page {p}** — ...{snip}...")
                        else:
                            st.info("No results found.")
                elif action == "summarize":
                    st.info("Summarization requested — not implemented yet.")
                else:
                    st.info("Unknown or unimplemented command.")

    st.markdown("---")
    st.caption("If the recorder doesn't show up, make sure you installed `streamlit-audiorecorder` and reload the page. For production, use HTTPS (some browsers restrict mic on insecure origins).")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    streamlit_app()
