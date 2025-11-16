#!/usr/bin/env python3
"""
Streamlit-only app for Voice Automated PDF Assistant (Feature 1 + Feature 2 in UI)

Usage:
    streamlit run app.py

Features:
 - Upload PDF
 - Per-page extraction (on-demand)
 - Next / Previous / Go-to navigation
 - Search across pages (show pages + snippets)
 - Read page using gTTS (MP3 generated server-side and played in-browser)
 - Voice command via audio file upload (WAV/OGG/MP3). The uploaded audio is recognized (Google Web Speech API)
   and parsed into commands which are executed (next/prev/goto/read/search/summarize scaffold).

Notes:
 - Heavy libraries are imported lazily to keep Streamlit startup fast.
 - MP3->WAV conversion requires pydub + ffmpeg; if missing, ask user to upload WAV.
"""

from __future__ import annotations
import os
import io
import re
from typing import List, Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("streamlit_voice_pdf")

# Utility: detect whether pydub/ffmpeg is available for mp3->wav conversion
def _has_pydub_and_ffmpeg() -> bool:
    try:
        import pydub  # type: ignore
        # pydub needs ffmpeg on PATH; a quick test:
        from pydub.utils import which  # type: ignore
        return which("ffmpeg") is not None or which("ffmpeg.exe") is not None
    except Exception:
        return False

# Lazy PDF extraction function
def extract_pages_from_bytes(pdf_bytes: bytes, use_pdfplumber: bool = True) -> List[str]:
    """
    Extract text from each page using pdfplumber if available, else PyPDF2.
    This function imports heavy libs only when executed.
    """
    texts: List[str] = []
    # Try pdfplumber first if requested
    if use_pdfplumber:
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text() or ""
                    texts.append(txt.strip())
            return texts
        except Exception as e:
            logger.debug("pdfplumber unavailable or failed: %s. Falling back to PyPDF2.", e)

    # PyPDF2 fallback
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i in range(len(reader.pages)):
            try:
                txt = reader.pages[i].extract_text() or ""
                texts.append(txt.strip())
            except Exception:
                texts.append("")
        return texts
    except Exception as e:
        logger.exception("PDF extraction failed: %s", e)
        raise RuntimeError(f"PDF extraction failed: {e}") from e

# Simple command parser (same logic as before)
def parse_command(recognized_text: str) -> Dict:
    if not recognized_text:
        return {"action": "none"}
    txt = recognized_text.lower().strip()
    if re.search(r"\b(next|next page|go to next)\b", txt):
        return {"action": "next"}
    if re.search(r"\b(previous|prev|previous page|go back)\b", txt):
        return {"action": "previous"}
    if re.search(r"\b(stop listening|stop|exit|quit)\b", txt):
        return {"action": "stop"}
    if re.search(r"\b(read (this )?page|read page|read)\b", txt):
        m = re.search(r"page (\d+)", txt)
        if m:
            return {"action": "read", "page": int(m.group(1)) - 1}
        return {"action": "read"}
    m = re.search(r"go to page (\d+)", txt)
    if m:
        return {"action": "goto", "page": int(m.group(1)) - 1}
    m2 = re.search(r"\b(go to|goto|open|page)\s+(\d+)\b", txt)
    if m2:
        return {"action":"goto", "page": int(m2.group(2)) - 1}
    m = re.search(r"(search for|find|look for|search)\s+(.*)", txt)
    if m:
        return {"action": "search", "query": m.group(2).strip()}
    if re.search(r"\b(summarize|summary|give me a summary|summarize this)\b", txt):
        return {"action": "summarize"}
    m = re.search(r"(highlight|mark)\s+(.*)", txt)
    if m:
        return {"action": "highlight", "text": m.group(2).strip()}
    m = re.search(r"(annotate|add note|note)\s+(.*)", txt)
    if m:
        return {"action": "annotate", "text": m.group(2).strip()}
    m = re.match(r"^(find|search|look)\s+(.*)", txt)
    if m:
        return {"action": "search", "query": m.group(2).strip()}
    m = re.search(r"page\s+(\d+)", txt)
    if m:
        return {"action": "goto", "page": int(m.group(1)) - 1}
    return {"action":"unknown", "text": recognized_text}

# TTS helper: uses gTTS to generate MP3 bytes and returns bytes
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
        logger.warning("gTTS generation failed: %s", e)
        return None

# Speech recognition from uploaded audio file -> returns recognized string or None
def recognize_audio_file(audio_bytes: bytes, filename: str) -> Optional[str]:
    """
    Accepts audio bytes and filename (to infer format). Returns recognized text using Google Web Speech API.
    If the audio is mp3 and pydub+ffmpeg available, converts to wav first.
    """
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        raise RuntimeError("SpeechRecognition not installed. Install `pip install SpeechRecognition` to use audio commands.")

    # if audio is mp3 and pydub+ffmpeg available, convert to wav bytes
    lower = filename.lower()
    wav_bytes = None
    if lower.endswith(".mp3") or lower.endswith(".m4a") or lower.endswith(".aac") or lower.endswith(".ogg"):
        if _has_pydub_and_ffmpeg():
            try:
                from pydub import AudioSegment  # type: ignore
                # pydub accepts bytes via BytesIO
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                out = io.BytesIO()
                audio.export(out, format="wav")
                out.seek(0)
                wav_bytes = out.read()
            except Exception as e:
                logger.warning("pydub conversion failed: %s", e)
                raise RuntimeError("Audio conversion failed. Ensure ffmpeg is installed or upload WAV.")
        else:
            raise RuntimeError("MP3/OGG audio uploaded but pydub+ffmpeg not available. Please install ffmpeg or upload WAV.")
    elif lower.endswith(".wav") or lower.endswith(".flac") or lower.endswith(".aiff") or lower.endswith(".aif"):
        wav_bytes = audio_bytes
    else:
        # unknown extension — try to pass as WAV
        wav_bytes = audio_bytes

    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = r.record(source)
            # use Google Web Speech (requires internet)
            text = r.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition request failed (probably no internet): {e}")
    except Exception as e:
        raise RuntimeError(f"Audio recognition failed: {e}")

# Streamlit UI
def streamlit_app():
    import streamlit as st  # local import for faster startup
    st.set_page_config(page_title="Voice PDF Assistant (Streamlit-only)", layout="wide")
    st.title("Voice Automated PDF Assistant — Streamlit-only")

    st.markdown("""
    **How to use (Streamlit-only):**
    1. Upload a PDF.  
    2. Use **Next / Previous / Go-to** to navigate pages.  
    3. Click **Read page (gTTS)** to hear the page (server generates MP3).  
    4. Use **Search** to find text across pages (shows snippets).  
    5. To give a voice command, record a short audio on your phone (or use browser recorder), download the audio, then upload it under *Voice command via audio file*. The app will transcribe and execute the command.
    """)

    st.sidebar.header("Settings")
    use_pdfplumber = st.sidebar.checkbox("Prefer pdfplumber extraction (if available)", value=True)
    tts_lang = st.sidebar.text_input("TTS language (gTTS)", value="en")
    max_snippet_chars = st.sidebar.number_input("Search snippet length", min_value=50, max_value=1500, value=300)

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is None:
        st.info("Upload a PDF to begin.")
        return

    # Read PDF bytes once
    pdf_bytes = uploaded_pdf.read()

    # Cache page-by-page extraction lazily: we will extract pages on demand and cache results.
    @st.cache_data(show_spinner=False)
    def cached_extract_all(pdf_b: bytes, use_pdfplumber_flag: bool):
        # returns list of page texts
        return extract_pages_from_bytes(pdf_b, use_pdfplumber=use_pdfplumber_flag)

    # We'll use lazy per-page content by extracting when needed but caching whole doc text list is easiest.
    with st.spinner("Extracting PDF text (first time only)..."):
        try:
            pages = cached_extract_all(pdf_bytes, use_pdfplumber)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            return

    num_pages = len(pages)
    st.success(f"Document loaded: {uploaded_pdf.name} — {num_pages} pages")

    # session state for current page index
    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0

    # Navigation controls
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1,1,1,2])
    with nav_col1:
        if st.button("Previous"):
            if st.session_state.page_idx > 0:
                st.session_state.page_idx -= 1
    with nav_col2:
        if st.button("Next"):
            if st.session_state.page_idx + 1 < num_pages:
                st.session_state.page_idx += 1
    with nav_col3:
        go_to = st.number_input("Go to page (1-indexed)", min_value=1, max_value=max(1,num_pages), value=st.session_state.page_idx+1)
        if st.button("Go"):
            st.session_state.page_idx = int(go_to)-1

    with nav_col4:
        if st.button("Read page (gTTS)"):
            text_to_read = pages[st.session_state.page_idx]
            if not text_to_read:
                st.info("No text extracted on this page to read.")
            else:
                mp3_bytes = generate_tts_mp3_bytes(text_to_read[:15000], lang=tts_lang)  # limit size for speed
                if mp3_bytes:
                    st.audio(mp3_bytes, format="audio/mp3")
                else:
                    st.info("gTTS not available or failed. Showing text instead.")
                    st.text_area("Page text", value=text_to_read, height=400)

    st.markdown("---")
    # Show current page text area (collapsed by default if long)
    page_text = pages[st.session_state.page_idx]
    st.subheader(f"Page {st.session_state.page_idx+1}/{num_pages}")
    if not page_text:
        st.info("This page appears to have no extracted text (likely scanned). Use OCR in CLI version.")
    st.text_area("Extracted page text", value=page_text, height=400)

    st.markdown("---")
    # Search section
    st.header("Search document")
    search_query = st.text_input("Enter search query and press Enter")
    if search_query:
        sq = search_query.lower().strip()
        found = []
        for i, txt in enumerate(pages):
            if txt and sq in txt.lower():
                # build snippet
                low = txt.lower()
                idx = low.find(sq)
                start = max(0, idx - 60)
                snippet = txt[start: start + max_snippet_chars]
                found.append((i+1, snippet.replace("\n", " ")))
        if found:
            st.success(f"Found on {len(found)} page(s): {[p for p,_ in found][:20]}")
            for p, snip in found[:50]:
                st.markdown(f"**Page {p}** — ...{snip}...")
                if st.button(f"Go to page {p}", key=f"goto_{p}"):
                    st.session_state.page_idx = p-1
        else:
            st.info("No results found.")

    st.markdown("---")
    # Voice command via audio upload
    st.header("Voice command via audio file (upload)")
    st.write("Record a short audio clip on your phone or using any browser recorder, download it (WAV/MP3), then upload here. The app will transcribe and execute commands like 'next page', 'read this page', 'go to page 5', 'search for introduction'.")
    audio_file = st.file_uploader("Upload audio command (WAV/MP3/OGG)", type=["wav","mp3","ogg","m4a","flac","aiff"])
    if audio_file is not None:
        st.info("Processing audio... (uses Google Web Speech API & requires internet).")
        audio_bytes = audio_file.read()
        try:
            recognized = recognize_audio_file(audio_bytes, audio_file.name)
        except Exception as e:
            st.error(f"Audio recognition failed: {e}")
            recognized = None
        if recognized is None:
            st.warning("Could not transcribe the audio (no recognized speech).")
        else:
            st.success(f"Recognized: {recognized}")
            # parse and execute
            cmd = parse_command(recognized)
            st.write("Parsed command:", cmd)
            action = cmd.get("action")
            if action == "next":
                if st.session_state.page_idx + 1 < num_pages:
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
            elif action in ("goto","read"):
                page = cmd.get("page")
                if page is None:
                    page = st.session_state.page_idx
                if page < 0 or page >= num_pages:
                    st.error("Requested page out of range.")
                else:
                    st.session_state.page_idx = page
                    if action == "read":
                        txt = pages[page]
                        if not txt:
                            st.info("No text on this page to read.")
                        else:
                            mp3_bytes = generate_tts_mp3_bytes(txt[:15000], lang=tts_lang)
                            if mp3_bytes:
                                st.audio(mp3_bytes, format="audio/mp3")
                            else:
                                st.text_area("Page text", value=txt, height=400)
                    else:
                        st.info(f"Moved to page {page+1}")
            elif action == "search":
                q = cmd.get("query","")
                if not q:
                    st.info("No search query detected.")
                else:
                    st.experimental_rerun()  # rerun to show search field (user can paste)
            elif action == "summarize":
                st.info("Summarization requested — not implemented yet in Streamlit version. Ask me to add summarization.")
            else:
                st.info("Command parsed but not implemented or unknown. Parsed: " + str(cmd))

    st.markdown("---")
    st.caption("This Streamlit-only UI implements PDF extraction, navigation, search, 'read page' via gTTS, and voice command via audio upload. For OCR, advanced TTS controls, or live mic capture from browser, we can add additional components (requires extra dependencies).")

# Run the app
if __name__ == "__main__":
    streamlit_app()
