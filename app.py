#!/usr/bin/env python3
"""
Streamlit-only Voice Automated PDF Assistant — All features combined + Summarization (Feature 6)

Usage:
    streamlit run app.py

Dependencies (recommended):
    pip install streamlit PyPDF2 pdfplumber SpeechRecognition gTTS streamlit-audiorecorder pydub pyttsx3

Notes:
 - Summaries are extractive (frequency-based) and run locally (no heavy model required).
 - Voice commands recognize "summarize page" and "summarize document" (or "summarize full").
"""
from __future__ import annotations
import io
import re
import os
import tempfile
import logging
from typing import List, Dict, Optional, Tuple
import difflib
import math
import heapq

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("voice_pdf_assistant_with_summary")

# ---------- optional components detection ----------
try:
    from streamlit_audiorecorder import audiorecorder  # type: ignore
    _HAS_AUDIOREC = True
except Exception:
    _HAS_AUDIOREC = False

# ---------- Navigation helper ----------
def go_to_page(page_number: int):
    """
    Set the session page index and attempt to rerun cleanly.
    Works across Streamlit versions (uses experimental_rerun if available).
    """
    st.session_state.page_idx = max(0, page_number - 1)
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass

# ---------- PDF extraction (lazy imports) ----------
def extract_pages_from_bytes(pdf_bytes: bytes, use_pdfplumber: bool = True) -> List[str]:
    """Extract text for each page using pdfplumber if available else PyPDF2."""
    texts: List[str] = []
    if use_pdfplumber:
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for p in pdf.pages:
                    texts.append((p.extract_text() or "").strip())
            return texts
        except Exception as e:
            logger.debug("pdfplumber not available/failed: %s; falling back to PyPDF2", e)
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

# ---------- TTS helpers ----------
def generate_gtts_mp3_bytes(text: str, lang: str = "en") -> Optional[bytes]:
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

def generate_pyttsx3_wav_bytes(text: str, rate: int = 160, volume: float = 1.0) -> Optional[bytes]:
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_name = tmp.name
        tmp.close()
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", rate)
        except Exception:
            pass
        try:
            engine.setProperty("volume", volume)
        except Exception:
            pass
        engine.save_to_file(text, tmp_name)
        engine.runAndWait()
        with open(tmp_name, "rb") as f:
            data = f.read()
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        return data
    except Exception as e:
        logger.exception("pyttsx3 TTS failed: %s", e)
        return None

@st.cache_data(show_spinner=False)
def cached_gtts(text: str, lang: str):
    return generate_gtts_mp3_bytes(text, lang=lang)

@st.cache_data(show_spinner=False)
def cached_pyttsx3(text: str, rate: int, volume: float):
    return generate_pyttsx3_wav_bytes(text, rate=rate, volume=volume)

# ---------- Speech recognition ----------
def recognize_wav_bytes(wav_bytes: bytes) -> Optional[str]:
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        raise RuntimeError("SpeechRecognition not installed. Install `pip install SpeechRecognition`.")
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = r.record(src)
            text = r.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition request failed (probably no internet): {e}")
    except Exception as e:
        raise RuntimeError(f"Audio recognition failed: {e}")

# ---------- Command parser (enhanced for summarize scope) ----------
def parse_command(recognized_text: str) -> Dict:
    """
    Returns dict with action and optional fields.
    For summarize: returns {"action":"summarize", "scope":"page"|"document"}
    """
    if not recognized_text:
        return {"action":"none"}
    txt = recognized_text.lower().strip()
    # navigation/read/search as before
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
    # summarize detection
    if re.search(r"\b(summarize|summary|summarize (page|document|full|pdf))\b", txt):
        # determine scope
        if re.search(r"(document|full|pdf)", txt):
            return {"action":"summarize", "scope":"document"}
        if re.search(r"page", txt):
            return {"action":"summarize", "scope":"page"}
        # default
        return {"action":"summarize", "scope":"page"}
    m = re.search(r"(search for|find|look for|search)\s+(.*)", txt)
    if m:
        return {"action":"search", "query": m.group(2).strip()}
    # fallback: short phrases become search
    if len(txt.split()) <= 6:
        return {"action":"search", "query": recognized_text.strip()}
    # annotate/highlight
    m = re.search(r"(highlight|mark)\s+(.*)", txt)
    if m:
        return {"action":"highlight", "text": m.group(2).strip()}
    m = re.search(r"(annotate|add note|note)\s+(.*)", txt)
    if m:
        return {"action":"annotate", "text": m.group(2).strip()}
    return {"action":"unknown", "text": recognized_text}

# ---------- Search utilities (unchanged) ----------
def find_query_in_pages(query: str, pages: List[str],
                        whole_word: bool=False,
                        case_sensitive: bool=False,
                        fuzzy: bool=False,
                        fuzzy_cutoff: float=0.6,
                        max_results: int=100) -> List[Tuple[int,str]]:
    results: List[Tuple[int,str]] = []
    if not query:
        return results
    q = query if case_sensitive else query.lower()
    for i, page_text in enumerate(pages):
        if not page_text:
            continue
        text_for_search = page_text if case_sensitive else page_text.lower()
        matched = False
        snippet = ""
        if whole_word:
            pattern = r"\b" + re.escape(q) + r"\b"
            m = re.search(pattern, text_for_search)
            if m:
                matched = True
                start = max(0, m.start()-60)
                snippet = page_text[start:start+300].replace("\n"," ")
        elif not fuzzy:
            if q in text_for_search:
                matched = True
                idx = text_for_search.find(q)
                start = max(0, idx-60)
                snippet = page_text[start:start+300].replace("\n"," ")
        else:
            words = re.findall(r"\w+", text_for_search)
            close = difflib.get_close_matches(q, words, n=5, cutoff=fuzzy_cutoff)
            if close:
                matched = True
                c = close[0]
                idx = text_for_search.find(c)
                start = max(0, idx-60)
                snippet = page_text[start:start+300].replace("\n"," ")
        if matched:
            results.append((i+1, snippet))
            if len(results) >= max_results:
                break
    return results

# ---------- Lightweight extractive summarizer ----------
# small english stopword list
_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each",
    "few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's","me","more","most","mustn't","my","myself",
    "no","nor","not",
    "of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to",
    "too",
    "under","until","up",
    "very",
    "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
}

def _split_sentences(text: str) -> List[str]:
    # naive sentence splitter that preserves punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def _tokenize_words(text: str) -> List[str]:
    # simple tokenizer: words only
    words = re.findall(r'\w+', text.lower())
    return words

@st.cache_data(show_spinner=False)
def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Extractive summary: score sentences by word frequency and pick top N.
    Returns joined sentences in original order.
    """
    if not text or len(text.strip()) < 20:
        return ""
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        # short text: return original
        return " ".join(sentences)

    words = _tokenize_words(text)
    # compute frequencies excluding stopwords and single chars
    freq: Dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS or len(w) <= 1:
            continue
        freq[w] = freq.get(w, 0) + 1
    if not freq:
        # fallback: return first N sentences
        return " ".join(sentences[:num_sentences])

    # normalize frequencies
    max_freq = max(freq.values())
    for k in freq:
        freq[k] = freq[k] / max_freq

    # score sentences
    sent_scores: List[Tuple[float,int,str]] = []  # (score, index, sentence)
    for i, s in enumerate(sentences):
        s_words = _tokenize_words(s)
        score = 0.0
        for w in s_words:
            if w in freq:
                score += freq[w]
        # penalize extremely short sentences
        length = len(s_words)
        if length == 0:
            continue
        # keep raw score
        sent_scores.append((score, i, s))
    # choose top num_sentences by score
    top = heapq.nlargest(num_sentences, sent_scores, key=lambda x: x[0])
    # sort by original order
    top_sorted = sorted(top, key=lambda x: x[1])
    summary_sentences = [t[2] for t in top_sorted]
    return " ".join(summary_sentences)

# cached full-document summarizer wrapper
@st.cache_data(show_spinner=False)
def summarize_pages_combined(pages: List[str], num_sentences: int = 5) -> str:
    combined = "\n\n".join([p for p in pages if p])
    return summarize_text(combined, num_sentences=num_sentences)

# ---------- Streamlit UI (all features + summarization) ----------
def streamlit_app():
    st.set_page_config(page_title="Voice PDF Assistant (All features + Summarize)", layout="wide")
    st.title("Voice Automated PDF Assistant — All features + Summarization")

    st.sidebar.header("Settings")
    use_pdfplumber = st.sidebar.checkbox("Prefer pdfplumber extraction (if available)", value=True)
    st.sidebar.markdown("**TTS settings**")
    tts_engine_choice = st.sidebar.selectbox("TTS engine", ["gTTS (online)", "pyttsx3 (offline)"])
    tts_lang = st.sidebar.text_input("gTTS language", value="en")
    pyttsx3_rate = st.sidebar.slider("pyttsx3 rate", min_value=80, max_value=300, value=160)
    pyttsx3_volume = st.sidebar.slider("pyttsx3 volume", min_value=0.0, max_value=1.0, value=1.0)

    st.sidebar.markdown("---")
    st.sidebar.header("Search options")
    whole_word = st.sidebar.checkbox("Whole-word match", value=False)
    case_sensitive = st.sidebar.checkbox("Case-sensitive", value=False)
    fuzzy = st.sidebar.checkbox("Fuzzy match", value=False)
    fuzzy_cutoff = st.sidebar.slider("Fuzzy cutoff", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    max_results = st.sidebar.number_input("Max results", min_value=1, max_value=200, value=100)

    st.markdown("## 1) Upload PDF")
    uploaded = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF to start. This app supports voice commands, TTS, search and summarization.")
        return

    pdf_bytes = uploaded.read()

    @st.cache_data(show_spinner=False)
    def cached_extract_all(b: bytes, use_pdfpl: bool):
        return extract_pages_from_bytes(b, use_pdfplumber=use_pdfpl)

    try:
        pages = cached_extract_all(pdf_bytes, use_pdfplumber)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return

    n_pages = len(pages)
    st.success(f"Loaded {uploaded.name} — {n_pages} pages")

    # session state
    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0

    # top navigation
    nav1, nav2, nav3, nav4 = st.columns([1,1,2,2])
    with nav1:
        if st.button("Previous"):
            if st.session_state.page_idx > 0:
                st.session_state.page_idx -= 1
    with nav2:
        if st.button("Next"):
            if st.session_state.page_idx + 1 < n_pages:
                st.session_state.page_idx += 1
    with nav3:
        goto_val = st.number_input("Go to page (1-indexed)", min_value=1, max_value=max(1,n_pages), value=st.session_state.page_idx+1)
        if st.button("Go"):
            st.session_state.page_idx = int(goto_val)-1
    with nav4:
        if st.button("Read page (TTS)"):
            txt = pages[st.session_state.page_idx]
            if not txt:
                st.info("No text on this page to read.")
            else:
                if tts_engine_choice.startswith("gTTS"):
                    mp3 = cached_gtts(txt[:30000], tts_lang)
                    if mp3:
                        st.audio(mp3, format="audio/mp3")
                    else:
                        st.error("gTTS unavailable or failed.")
                else:
                    wav = cached_pyttsx3(txt[:8000], pyttsx3_rate, pyttsx3_volume)
                    if wav:
                        st.audio(wav, format="audio/wav")
                    else:
                        st.error("pyttsx3 unavailable or failed.")

    st.markdown("---")
    st.subheader(f"Page {st.session_state.page_idx+1}/{n_pages}")
    current_text = pages[st.session_state.page_idx] or ""
    if not current_text:
        st.info("This page appears to have no extracted text (likely scanned).")
    st.text_area("Page text", value=current_text, height=300)

    st.markdown("---")
    # Summarization UI
    st.header("Summarization (Feature 6)")
    sum_col_page, sum_col_doc = st.columns([1,1])
    with sum_col_page:
        st.subheader("Summarize current page")
        page_sentences = st.number_input("Sentences for page summary", min_value=1, max_value=10, value=3)
        if st.button("Generate page summary"):
            with st.spinner("Summarizing page..."):
                summary = summarize_text(current_text, num_sentences=page_sentences)
            if not summary:
                st.info("Page too short or no text to summarize. Showing page text instead.")
                st.text_area("Page text", value=current_text, height=200)
            else:
                st.success("Page summary:")
                st.write(summary)
                # TTS play/download options
                cols = st.columns([1,1,2])
                with cols[0]:
                    if st.button("Play page summary (TTS)", key="play_page_summary"):
                        if tts_engine_choice.startswith("gTTS"):
                            mp3 = cached_gtts(summary[:30000], tts_lang)
                            if mp3:
                                st.audio(mp3, format="audio/mp3")
                            else:
                                st.error("gTTS failed.")
                        else:
                            wav = cached_pyttsx3(summary[:5000], pyttsx3_rate, pyttsx3_volume)
                            if wav:
                                st.audio(wav, format="audio/wav")
                            else:
                                st.error("pyttsx3 failed.")
                with cols[1]:
                    if st.button("Download page summary (audio)", key="download_page_summary"):
                        if tts_engine_choice.startswith("gTTS"):
                            mp3 = cached_gtts(summary[:30000], tts_lang)
                            if mp3:
                                st.download_button("Download MP3", data=mp3, file_name=f"page_{st.session_state.page_idx+1}_summary.mp3", mime="audio/mpeg")
                            else:
                                st.error("gTTS failed.")
                        else:
                            wav = cached_pyttsx3(summary[:5000], pyttsx3_rate, pyttsx3_volume)
                            if wav:
                                st.download_button("Download WAV", data=wav, file_name=f"page_{st.session_state.page_idx+1}_summary.wav", mime="audio/wav")
                            else:
                                st.error("pyttsx3 failed.")
                with cols[2]:
                    st.download_button("Download page summary (text)", data=summary, file_name=f"page_{st.session_state.page_idx+1}_summary.txt", mime="text/plain")

    with sum_col_doc:
        st.subheader("Summarize full document")
        doc_sentences = st.number_input("Sentences for document summary", min_value=1, max_value=50, value=7)
        if st.button("Generate document summary"):
            with st.spinner("Summarizing document..."):
                full_summary = summarize_pages_combined(pages, num_sentences=doc_sentences)
            if not full_summary:
                st.info("Document has insufficient text to summarize.")
            else:
                st.success("Document summary:")
                st.write(full_summary)
                cols2 = st.columns([1,1,2])
                with cols2[0]:
                    if st.button("Play doc summary (TTS)", key="play_doc_summary"):
                        if tts_engine_choice.startswith("gTTS"):
                            mp3 = cached_gtts(full_summary[:30000], tts_lang)
                            if mp3:
                                st.audio(mp3, format="audio/mp3")
                            else:
                                st.error("gTTS failed.")
                        else:
                            wav = cached_pyttsx3(full_summary[:5000], pyttsx3_rate, pyttsx3_volume)
                            if wav:
                                st.audio(wav, format="audio/wav")
                            else:
                                st.error("pyttsx3 failed.")
                with cols2[1]:
                    if st.button("Download doc summary (audio)", key="download_doc_summary"):
                        if tts_engine_choice.startswith("gTTS"):
                            mp3 = cached_gtts(full_summary[:30000], tts_lang)
                            if mp3:
                                st.download_button("Download MP3", data=mp3, file_name=f"{uploaded.name}_summary.mp3", mime="audio/mpeg")
                            else:
                                st.error("gTTS failed.")
                        else:
                            wav = cached_pyttsx3(full_summary[:5000], pyttsx3_rate, pyttsx3_volume)
                            if wav:
                                st.download_button("Download WAV", data=wav, file_name=f"{uploaded.name}_summary.wav", mime="audio/wav")
                            else:
                                st.error("pyttsx3 failed.")
                with cols2[2]:
                    st.download_button("Download doc summary (text)", data=full_summary, file_name=f"{uploaded.name}_summary.txt", mime="text/plain")

    st.markdown("---")
    # Search panel (typed)
    st.header("Search (typed)")
    query = st.text_input("Type a query and press Enter")
    if query:
        with st.spinner("Searching..."):
            results = find_query_in_pages(query, pages, whole_word=whole_word, case_sensitive=case_sensitive, fuzzy=fuzzy, fuzzy_cutoff=fuzzy_cutoff, max_results=max_results)
        if not results:
            st.info("No results found.")
        else:
            st.success(f"Found on {len(results)} page(s): {[p for p,_ in results][:20]}")
            for p, snip in results:
                c0, c1, c2 = st.columns([1,6,1])
                c0.markdown(f"**Page {p}**")
                c1.write("..."+snip+"...")
                if c2.button("Go to page", key=f"go_{p}"):
                    go_to_page(p)

    st.markdown("---")
    # Voice recorder (browser mic) and audio file fallback
    st.header("Voice command (browser mic) — Start / Stop to capture a short command (1-8s)")
    if not _HAS_AUDIOREC:
        st.warning("Browser mic recorder component (`streamlit-audiorecorder`) not installed. Install with: pip install streamlit-audiorecorder")
        st.info("Fallback: use the 'Upload audio file for commands' below.")
    else:
        st.write("Press Start then Stop after speaking a command (e.g., 'next page', 'go to page 5', 'summarize page', 'summarize document').")
        wav_bytes = audiorecorder(duration=8, key="recorder_all")
        if wav_bytes:
            st.success("Recording captured — transcribing...")
            try:
                recognized = recognize_wav_bytes(wav_bytes)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                recognized = None
            if recognized is None:
                st.warning("Could not transcribe audio.")
            else:
                st.success(f"Recognized: {recognized}")
                cmd = parse_command(recognized)
                st.write("Parsed command:", cmd)
                action = cmd.get("action")
                if action == "summarize":
                    scope = cmd.get("scope","page")
                    if scope == "document":
                        with st.spinner("Summarizing document..."):
                            full_summary = summarize_pages_combined(pages, num_sentences=7)
                        if not full_summary:
                            st.info("Document has insufficient text to summarize.")
                        else:
                            st.success("Document summary:")
                            st.write(full_summary)
                    else:
                        with st.spinner("Summarizing page..."):
                            page_summary = summarize_text(pages[st.session_state.page_idx], num_sentences=3)
                        if not page_summary:
                            st.info("Page too short to summarize.")
                        else:
                            st.success("Page summary:")
                            st.write(page_summary)
                # existing actions: next, previous, goto, read, search, etc.
                elif action == "next":
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
                elif action in ("goto","read"):
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
                                st.info("No text to read on this page.")
                            else:
                                if tts_engine_choice.startswith("gTTS"):
                                    b = cached_gtts(txt[:30000], tts_lang)
                                    if b:
                                        st.audio(b, format="audio/mp3")
                                    else:
                                        st.error("gTTS failed.")
                                else:
                                    b = cached_pyttsx3(txt[:8000], pyttsx3_rate, pyttsx3_volume)
                                    if b:
                                        st.audio(b, format="audio/wav")
                                    else:
                                        st.error("pyttsx3 failed.")
                        else:
                            st.info(f"Moved to page {page+1}")
                elif action == "search":
                    q = cmd.get("query","")
                    if not q:
                        st.info("No search query detected.")
                    else:
                        with st.spinner("Searching..."):
                            results = find_query_in_pages(q, pages, whole_word=whole_word, case_sensitive=case_sensitive, fuzzy=fuzzy, fuzzy_cutoff=fuzzy_cutoff, max_results=max_results)
                        if not results:
                            st.info(f"No results for '{q}'.")
                        else:
                            st.success(f"Found '{q}' on {len(results)} page(s): {[p for p,_ in results][:20]}")
                            for p, snip in results:
                                r0, r1, r2 = st.columns([1,6,1])
                                r0.markdown(f"**Page {p}**")
                                r1.write("..."+snip+"...")
                                if r2.button("Go to page", key=f"vgo_{p}"):
                                    go_to_page(p)
                else:
                    st.info("Unknown or unimplemented command.")

    st.markdown("---")
    st.header("Upload audio file for commands (fallback)")
    audio_file = st.file_uploader("Upload WAV/MP3/OGG file", type=["wav","mp3","ogg","m4a","flac","aiff"])
    if audio_file is not None:
        st.info("Processing uploaded audio...")
        audio_bytes = audio_file.read()
        fname = audio_file.name.lower()
        wav_bytes2 = None
        if fname.endswith(".mp3") or fname.endswith(".ogg") or fname.endswith(".m4a"):
            try:
                from pydub import AudioSegment  # type: ignore
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                out = io.BytesIO()
                audio.export(out, format="wav")
                out.seek(0)
                wav_bytes2 = out.read()
            except Exception as e:
                st.error("Conversion failed. Install pydub+ffmpeg or upload WAV.")
                wav_bytes2 = None
        else:
            wav_bytes2 = audio_bytes

        if wav_bytes2:
            try:
                recognized = recognize_wav_bytes(wav_bytes2)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                recognized = None
            if recognized is None:
                st.warning("Could not transcribe uploaded audio.")
            else:
                st.success(f"Recognized: {recognized}")
                cmd = parse_command(recognized)
                st.write("Parsed command:", cmd)
                action = cmd.get("action")
                if action == "summarize":
                    scope = cmd.get("scope","page")
                    if scope == "document":
                        with st.spinner("Summarizing document..."):
                            full_summary = summarize_pages_combined(pages, num_sentences=7)
                        if not full_summary:
                            st.info("Document has insufficient text to summarize.")
                        else:
                            st.success("Document summary:")
                            st.write(full_summary)
                    else:
                        with st.spinner("Summarizing page..."):
                            page_summary = summarize_text(pages[st.session_state.page_idx], num_sentences=3)
                        if not page_summary:
                            st.info("Page too short to summarize.")
                        else:
                            st.success("Page summary:")
                            st.write(page_summary)
                # reuse other actions
                elif action == "next":
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
                elif action in ("goto","read"):
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
                                if tts_engine_choice.startswith("gTTS"):
                                    b = cached_gtts(txt[:30000], tts_lang)
                                    if b:
                                        st.audio(b, format="audio/mp3")
                                    else:
                                        st.error("gTTS failed.")
                                else:
                                    b = cached_pyttsx3(txt[:8000], pyttsx3_rate, pyttsx3_volume)
                                    if b:
                                        st.audio(b, format="audio/wav")
                                    else:
                                        st.error("pyttsx3 failed.")
                        else:
                            st.info(f"Moved to page {page+1}")
                elif action == "search":
                    q = cmd.get("query","")
                    if not q:
                        st.info("No search query detected.")
                    else:
                        with st.spinner("Searching..."):
                            results = find_query_in_pages(q, pages, whole_word=whole_word, case_sensitive=case_sensitive, fuzzy=fuzzy, fuzzy_cutoff=fuzzy_cutoff, max_results=max_results)
                        if not results:
                            st.info(f"No results for '{q}'.")
                        else:
                            st.success(f"Found '{q}' on {len(results)} page(s).")
                            for p, snip in results:
                                s0, s1, s2 = st.columns([1,6,1])
                                s0.markdown(f"**Page {p}**")
                                s1.write("..."+snip+"...")
                                if s2.button("Go to page", key=f"ugo_{p}"):
                                    go_to_page(p)
                else:
                    st.info("Unknown or unimplemented command from uploaded audio.")

    st.markdown("---")
    st.caption("Summaries are extractive and local. If you want an abstractive (neural) summarizer integrated (e.g., transformer/Whisper+LLM), I can add it next — it requires additional packages and compute.")

# Entrypoint
if __name__ == "__main__":
    streamlit_app()
