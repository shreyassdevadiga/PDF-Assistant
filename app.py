#!/usr/bin/env python3
"""
Streamlit-only Voice PDF Assistant — Browser Mic recorder + TTS controls (Feature 1-3)

Requirements:
    pip install streamlit PyPDF2 pdfplumber SpeechRecognition gTTS streamlit-audiorecorder pyttsx3

Run:
    streamlit run app.py

Notes:
 - gTTS requires internet and produces MP3 audio.
 - pyttsx3 works offline and supports rate/volume but must be installed in the same environment.
 - The in-browser recorder uses streamlit-audiorecorder; if not installed the app shows instructions.
"""
from __future__ import annotations
import io
import re
import tempfile
import os
import logging
from typing import List, Dict, Optional

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("streamlit_voice_pdf_tts")

# Try to import the browser recorder component
try:
    from streamlit_audiorecorder import audiorecorder  # type: ignore
    _HAS_AUDIOREC = True
except Exception:
    _HAS_AUDIOREC = False

# ---------- PDF extraction ----------
def extract_pages_from_bytes(pdf_bytes: bytes, use_pdfplumber: bool = True) -> List[str]:
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
    """Generate MP3 bytes using gTTS (requires internet)."""
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

def generate_pyttsx3_audio_bytes(text: str, rate: int = 160, volume: float = 1.0, fmt: str = "wav") -> Optional[bytes]:
    """
    Use pyttsx3 to generate audio file (WAV). Returns bytes or None if pyttsx3 not available.
    Note: pyttsx3.save_to_file -> engine.runAndWait will write to disk; we read and return bytes.
    """
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return None
    try:
        # write to a temporary file, then read bytes
        suffix = ".wav" if fmt == "wav" else ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
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
        # save_to_file supports wav on many platforms
        engine.save_to_file(text, tmp_name)
        engine.runAndWait()
        # read bytes
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

# Caching wrapper for TTS generation to avoid regeneration on repeated play
@st.cache_data(show_spinner=False)
def cached_gtts(text: str, lang: str) -> Optional[bytes]:
    return generate_gtts_mp3_bytes(text, lang=lang)

@st.cache_data(show_spinner=False)
def cached_pyttsx3(text: str, rate: int, volume: float) -> Optional[bytes]:
    return generate_pyttsx3_audio_bytes(text, rate=rate, volume=volume, fmt="wav")

# ---------- Command parser (same as before) ----------
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

# ---------- Recognize WAV bytes using SpeechRecognition ----------
def recognize_wav_bytes(wav_bytes: bytes) -> Optional[str]:
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        raise RuntimeError("SpeechRecognition not installed. Install with `pip install SpeechRecognition`.")
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

# ---------- Streamlit app ----------
def streamlit_app():
    st.set_page_config(page_title="Voice PDF Assistant (TTS + Browser Mic)", layout="wide")
    st.title("Voice Automated PDF Assistant — TTS Controls & Browser Mic")

    st.markdown(
        "Upload a PDF, navigate pages, record a short command with your browser mic, "
        "and use the TTS panel to play or download audio of page text or custom text."
    )

    # Sidebar settings
    st.sidebar.header("Settings")
    use_pdfplumber = st.sidebar.checkbox("Prefer pdfplumber extraction (if available)", value=True)
    tts_engine_choice = st.sidebar.radio("TTS engine", options=["gTTS (online)", "pyttsx3 (offline, if installed)"], index=0)
    tts_lang = st.sidebar.text_input("gTTS language code (e.g. 'en', 'hi')", value="en")
    pyttsx3_rate = st.sidebar.slider("pyttsx3 rate (words per minute)", min_value=80, max_value=300, value=160)
    pyttsx3_volume = st.sidebar.slider("pyttsx3 volume", min_value=0.0, max_value=1.0, value=1.0)

    # Load PDF
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

    # page navigation
    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0

    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        if st.button("Previous"):
            if st.session_state.page_idx > 0:
                st.session_state.page_idx -= 1
    with c2:
        if st.button("Next"):
            if st.session_state.page_idx + 1 < n_pages:
                st.session_state.page_idx += 1
    with c3:
        goto_val = st.number_input("Go to page (1-indexed)", min_value=1, max_value=max(1,n_pages), value=st.session_state.page_idx+1)
        if st.button("Go"):
            st.session_state.page_idx = int(goto_val)-1
    # TTS quick read button moved a bit later into TTS panel

    st.markdown("---")
    st.subheader(f"Page {st.session_state.page_idx+1}/{n_pages}")
    page_text = pages[st.session_state.page_idx]
    if not page_text:
        st.info("This page appears to have no extracted text (likely scanned).")
    st.text_area("Extracted page text", value=page_text, height=320)

    st.markdown("---")
    st.header("Text-to-Speech (TTS) controls")

    # TTS selection: what to read
    t1, t2 = st.columns([2,1])
    with t1:
        tts_target = st.selectbox("TTS source", options=["Current page", "Selection (enter below)", "Custom text"])
        if tts_target == "Selection (enter below)":
            selection_text = st.text_area("Enter the text selection to read", value="", height=120)
        elif tts_target == "Custom text":
            custom_text = st.text_area("Enter custom text to read", value="", height=120)
        else:
            # current page
            selection_text = ""
            custom_text = ""
    with t2:
        st.markdown("**Engine settings**")
        st.write(f"Selected engine: **{tts_engine_choice}**")
        if tts_engine_choice.startswith("gTTS"):
            st.info("gTTS is online and produces high-quality MP3 audio. Rate slider is not applied to gTTS.")
        else:
            # check pyttsx3 availability
            try:
                import pyttsx3  # type: ignore
                _HAS_PYTTSX3 = True
            except Exception:
                _HAS_PYTTSX3 = False
            if not _HAS_PYTTSX3:
                st.warning("pyttsx3 not installed in this environment. Install `pip install pyttsx3` to use offline TTS.")
            else:
                st.write(f"Rate: {pyttsx3_rate}, Volume: {pyttsx3_volume}")

    # Build the final text to speak
    if tts_target == "Selection (enter below)":
        text_to_speak = selection_text.strip()
    elif tts_target == "Custom text":
        text_to_speak = custom_text.strip()
    else:
        text_to_speak = page_text.strip()

    if not text_to_speak:
        st.info("No text available for TTS. Provide custom text or ensure page has extracted text.")
    else:
        # TTS engine selection and play/download buttons
        cols = st.columns([1,1,1,2])
        with cols[0]:
            if st.button("Play TTS"):
                # choose engine
                if tts_engine_choice.startswith("gTTS"):
                    mp3_bytes = cached_gtts(text_to_speak[:30000], tts_lang)  # limit size to keep speed
                    if mp3_bytes:
                        st.audio(mp3_bytes, format="audio/mp3")
                    else:
                        st.error("gTTS not available or failed. Install `gtts` and ensure internet connectivity.")
                else:
                    # pyttsx3 path
                    try:
                        import pyttsx3  # type: ignore
                    except Exception:
                        st.error("pyttsx3 not installed. Install with `pip install pyttsx3` to use offline TTS.")
                        mp3_bytes = None
                    else:
                        wav_bytes = cached_pyttsx3(text_to_speak[:5000], pyttsx3_rate, pyttsx3_volume)
                        if wav_bytes:
                            # pyttsx3 produced WAV bytes: play in browser
                            st.audio(wav_bytes, format="audio/wav")
                        else:
                            st.error("pyttsx3 TTS generation failed.")
        with cols[1]:
            if st.button("Download TTS"):
                # generate bytes then download
                if tts_engine_choice.startswith("gTTS"):
                    mp3_bytes = cached_gtts(text_to_speak[:30000], tts_lang)
                    if mp3_bytes:
                        st.download_button("Download MP3", data=mp3_bytes, file_name=f"tts_{st.session_state.page_idx+1}.mp3", mime="audio/mpeg")
                    else:
                        st.error("gTTS failed. Install gTTS or try again later.")
                else:
                    try:
                        import pyttsx3  # type: ignore
                    except Exception:
                        st.error("pyttsx3 not installed. Install with `pip install pyttsx3`.")
                    else:
                        wav_bytes = cached_pyttsx3(text_to_speak[:5000], pyttsx3_rate, pyttsx3_volume)
                        if wav_bytes:
                            st.download_button("Download WAV", data=wav_bytes, file_name=f"tts_{st.session_state.page_idx+1}.wav", mime="audio/wav")
                        else:
                            st.error("pyttsx3 generation failed.")
        with cols[2]:
            # Quick preset buttons to read current page or a small summary
            if st.button("Read current page (quick)"):
                t = page_text[:800] or "This page has no extracted text."
                if tts_engine_choice.startswith("gTTS"):
                    b = cached_gtts(t, tts_lang)
                    if b:
                        st.audio(b, format="audio/mp3")
                    else:
                        st.error("gTTS failed.")
                else:
                    b = cached_pyttsx3(t, pyttsx3_rate, pyttsx3_volume)
                    if b:
                        st.audio(b, format="audio/wav")
                    else:
                        st.error("pyttsx3 failed.")
        with cols[3]:
            st.caption("Tip: Use Play to preview audio. Use Download to save MP3/WAV to device.")

    st.markdown("---")
    st.header("Record voice command (browser mic)")
    if not _HAS_AUDIOREC:
        st.warning("`streamlit-audiorecorder` is not installed. Install with:\n\npip install streamlit-audiorecorder\n\nAfter install, reload this page.")
        st.info("Fallback: you can upload an audio file for commands.")
    else:
        st.write("Press Start / Stop to record a short voice command (1-8 seconds).")
        wav_bytes = audiorecorder(duration=8, key="recorder")
        if wav_bytes:
            st.success("Recording captured. Transcribing...")
            try:
                recognized = recognize_wav_bytes(wav_bytes)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                recognized = None
            if not recognized:
                st.warning("Could not transcribe the audio.")
            else:
                st.success(f"Recognized: {recognized}")
                cmd = parse_command(recognized)
                st.write("Parsed command:", cmd)
                action = cmd.get("action")
                # Execute parsed command (navigation/read/search)
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
                                # play TTS according to selected engine
                                if tts_engine_choice.startswith("gTTS"):
                                    b = cached_gtts(txt[:30000], tts_lang)
                                    if b:
                                        st.audio(b, format="audio/mp3")
                                    else:
                                        st.error("gTTS failed.")
                                else:
                                    b = cached_pyttsx3(txt[:5000], pyttsx3_rate, pyttsx3_volume)
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
                        sq = q.lower().strip()
                        found = []
                        for i, txt in enumerate(pages):
                            if txt and sq in txt.lower():
                                idx = txt.lower().find(sq)
                                start = max(0, idx-60)
                                snippet = txt[start:start+300].replace("\n"," ")
                                found.append((i+1, snippet))
                        if found:
                            st.success(f"Found on {len(found)} page(s): {[p for p,_ in found]}")
                            for p, snip in found[:50]:
                                st.markdown(f"**Page {p}** — ...{snip}...")
                        else:
                            st.info("No results found.")
                elif action == "summarize":
                    st.info("Summarization requested — not yet implemented.")
                else:
                    st.info("Unknown or unimplemented command.")

    st.markdown("---")
    st.header("Fallback: upload audio file (WAV/MP3) for commands")
    audio_file = st.file_uploader("Upload audio command file", type=["wav","mp3","ogg","m4a","flac","aiff"])
    if audio_file is not None:
        st.info("Processing uploaded audio.")
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
                st.error("MP3/OGG conversion failed. Install pydub+ffmpeg or upload WAV.")
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
                # execute same logic as above (navigate/read/search)
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
                                if tts_engine_choice.startswith("gTTS"):
                                    b = cached_gtts(txt[:30000], tts_lang)
                                    if b:
                                        st.audio(b, format="audio/mp3")
                                    else:
                                        st.error("gTTS failed.")
                                else:
                                    b = cached_pyttsx3(txt[:5000], pyttsx3_rate, pyttsx3_volume)
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
                        sq = q.lower().strip()
                        found = []
                        for i, txt in enumerate(pages):
                            if txt and sq in txt.lower():
                                idx = txt.lower().find(sq)
                                start = max(0, idx-60)
                                snippet = txt[start:start+300].replace("\n"," ")
                                found.append((i+1, snippet))
                        if found:
                            st.success(f"Found on {len(found)} page(s): {[p for p,_ in found]}")
                            for p, snip in found[:50]:
                                st.markdown(f"**Page {p}** — ...{snip}...")
                        else:
                            st.info("No results found.")
                else:
                    st.info("Unknown or unimplemented command.")

    st.markdown("---")
    st.caption("TTS: gTTS requires internet (online). pyttsx3 is offline but must be installed in the same environment. "
               "Use 'Play' to preview audio and 'Download' to save files.")

# Entrypoint
if __name__ == "__main__":
    streamlit_app()
