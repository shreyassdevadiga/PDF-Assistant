# app_streamlit.py
"""
Streamlit voice-driven PDF assistant (adapted from user's script).
- Runs the voice assistant on the Streamlit server in a background thread.
- Server must have microphone & speakers available to Python process.
"""
import os
import re
import time
import threading
import queue
import traceback

import streamlit as st
import base64
import PyPDF2
import pyttsx3
import speech_recognition as sr

# Attempt optional imports (lazy, may not be installed)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    try:
        # older/newer langchain samename loader
        from langchain_community.document_loaders import PyPDFLoader
    except Exception:
        from langchain_community.document_loaders import PyPDFLoader
except Exception:
    RecursiveCharacterTextSplitter = None
    PyPDFLoader = None

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    T5Tokenizer = None
    T5ForConditionalGeneration = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

# ---------- Config ----------
DATA_DIR = "data"
MODEL_CHECKPOINT = "LaMini-Flan-T5-248M"  # kept as your checkpoint; may need full repo id depending on availability
ASR_TIMEOUT = 8
ASR_PHRASE_LIMIT = 6
LOG_Q = queue.Queue(maxsize=500)

# ---------- Utilities ----------
def log(msg: str):
    """Thread-safe log to streamlit UI queue."""
    try:
        LOG_Q.put_nowait(f"{time.strftime('%H:%M:%S')} — {msg}")
    except queue.Full:
        pass

def drain_logs():
    lines = []
    while not LOG_Q.empty():
        try:
            lines.append(LOG_Q.get_nowait())
        except queue.Empty:
            break
    return lines

# ---------- TTS (single engine) ----------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)

def speak(text: str, block: bool = True):
    if not text:
        return
    try:
        tts_engine.say(str(text))
        if block:
            tts_engine.runAndWait()
        else:
            threading.Thread(target=tts_engine.runAndWait, daemon=True).start()
    except Exception as e:
        log(f"[TTS ERROR] {e}")

# ---------- Speech recognition ----------
recognizer = sr.Recognizer()

def listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT):
    """
    Listens from server microphone and returns recognized lowercase text.
    Returns None for silence / unknown / errors.
    """
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            return text.lower().strip()
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        log(f"[ASR RequestError] {e}")
        speak("Speech recognition service error.")
        return None
    except Exception as e:
        log(f"[ASR Exception] {e}")
        return None

# ---------- PDF helpers ----------
def displayPDF(file):
    """Embed PDF in Streamlit page."""
    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        return pdf_display
    except Exception as e:
        log(f"[PDF preview error] {e}")
        return None

def extract_text_from_pdf(filepath):
    """Extract all text from a PDF using PyPDF2 (returns concatenated string)."""
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                page_text = p.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        log(f"[PDF extract error] {e}")
    return text

def search_word_in_pdf(filepath, search_word):
    try:
        reader = PyPDF2.PdfReader(open(filepath, "rb"))
        total_pages = len(reader.pages)
        occurrences = []
        for i in range(total_pages):
            page_text = reader.pages[i].extract_text() or ""
            if search_word.lower() in page_text.lower():
                occurrences.append(i + 1)
        return occurrences
    except Exception as e:
        log(f"[PDF search error] {e}")
        return []

# ---------- Langchain / LLM helpers (optional) ----------
def file_preprocessing(file):
    # Use PyPDFLoader + RecursiveCharacterTextSplitter if available, otherwise fallback
    if PyPDFLoader and RecursiveCharacterTextSplitter:
        try:
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            texts = splitter.split_documents(pages)
            final = ""
            for t in texts:
                final += t.page_content + "\n"
            return final
        except Exception as e:
            log(f"[file_preprocessing error] {e}")
    # fallback:
    return extract_text_from_pdf(file)

# Lazy LLM variables
_llm_pipeline = None
_tokenizer = None
_base_model = None
_llm_lock = threading.Lock()

def load_llm_if_needed():
    global _llm_pipeline, _tokenizer, _base_model
    with _llm_lock:
        if _llm_pipeline is not None:
            return True
        if not TRANSFORMERS_AVAILABLE:
            log("Transformers not available; LLM disabled.")
            return False
        try:
            log("Loading tokenizer & model (this may take time)...")
            _tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
            _base_model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
            _llm_pipeline = pipeline("summarization", model=_base_model, tokenizer=_tokenizer)
            log("LLM loaded.")
            return True
        except Exception as e:
            log(f"[LLM load error] {e}")
            return False

def llm_summarize(filepath):
    if not load_llm_if_needed():
        return "Summarization model not available."
    text = file_preprocessing(filepath)
    if len(text) > 30000:
        text = text[:30000]
    try:
        out = _llm_pipeline(text, max_length=300, min_length=30)
        return out[0].get("summary_text", "")
    except Exception as e:
        log(f"[LLM summarize error] {e}")
        return "Summarization failed."

def answer_query_t5(query, text):
    if not load_llm_if_needed():
        return "Q&A model not available."
    try:
        prompt = f"question: {query} context: {text}"
        inputs = _tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        answer_ids = _base_model.generate(inputs, max_length=200, num_return_sequences=1, num_beams=3, early_stopping=True)
        answer = _tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        log(f"[LLM Q/A error] {e}")
        return "Failed to generate answer."

# ---------- Voice command parsing & reader ----------
def parse_page_number_from_command(command):
    nums = re.findall(r'\d+', command or "")
    if nums:
        try:
            return int(nums[0])
        except:
            return None
    return None

def read_page(pdf_reader, page_number, recognizer):
    # page_number expected 0-based
    try:
        page = pdf_reader.pages[page_number]
        text = page.extract_text() or ""
    except Exception:
        speak("Unable to read that page.")
        return
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    for s in sentences:
        speak(s)
        # quick non-blocking check for "stop"
        quick = listen(timeout=0.6, phrase_time_limit=1)
        if quick and "stop" in quick:
            speak("Stopping reading.")
            return

# ---------- Assistant loop (runs in background) ----------
def assistant_loop():
    try:
        speak("Hello, I am SpeakLink. I will listen for a filename in the data folder.")
        log("Assistant started. Asking for filename.")
        # announce data files if any
        os.makedirs(DATA_DIR, exist_ok=True)
        pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
        if pdfs:
            speak(f"I see {len(pdfs)} files in the data folder.")
            for fn in pdfs:
                speak(fn.replace(".pdf", "").replace("_", " "), block=False)
                time.sleep(0.05)
        else:
            speak("No PDF files in the data folder. Please add files to data/ before using voice filename.")

        # listen for filename
        attempts = 0
        filepath = None
        while attempts < 6 and not filepath:
            attempts += 1
            speak("Please say the filename without the .pdf extension, or say quit to exit.")
            heard = listen(timeout=8, phrase_time_limit=5)
            if not heard:
                speak("I didn't hear anything. Please repeat.")
                continue
            log(f"Filename heard: {heard}")
            if any(x in heard for x in ["quit", "exit", "stop"]):
                speak("Exiting assistant.")
                log("Exit received while waiting for filename.")
                return
            # tolerate 'underscore' and 'space'
            candidate = heard.replace("underscore", "_").replace("dash", "-").replace("space", " ").strip()
            variants = [
                os.path.join(DATA_DIR, candidate + ".pdf"),
                os.path.join(DATA_DIR, candidate.replace(" ", "") + ".pdf"),
                os.path.join(DATA_DIR, candidate.replace(" ", "_") + ".pdf"),
            ]
            chosen = None
            for v in variants:
                if os.path.exists(v):
                    chosen = v
                    break
            if chosen:
                filepath = chosen
                speak(f"Opening {os.path.basename(chosen)}.")
                log(f"Selected file: {chosen}")
                break
            else:
                speak(f"Could not find {candidate} in data folder.")
                if attempts == 2 and pdfs:
                    speak("Available files are:")
                    for fn in pdfs:
                        speak(fn.replace(".pdf", "").replace("_", " "), block=False)
                continue

        if not filepath:
            speak("No file chosen. Exiting assistant.")
            log("No filepath selected; assistant exiting.")
            return

        # display PDF preview in Streamlit through log (UI will refresh)
        log(f"Loaded file: {filepath}")

        # open pdf
        try:
            fhandle = open(filepath, "rb")
            pdf_reader = PyPDF2.PdfReader(fhandle)
            total_pages = len(pdf_reader.pages)
            log(f"PDF has {total_pages} pages.")
        except Exception as e:
            log(f"[PDF open error] {e}")
            speak("Failed to open PDF.")
            return

        # optionally start loading LLM in background if available
        if TRANSFORMERS_AVAILABLE:
            threading.Thread(target=load_llm_if_needed, daemon=True).start()

        # main command loop
        querying = False
        while True:
            speak("Listening for a command.")
            cmd = listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT)
            if not cmd:
                log("No command heard; retrying.")
                time.sleep(0.3)
                continue

            log(f"Command: {cmd}")
            # EXIT
            if any(x in cmd for x in ["exit", "quit", "speaklink exit", "speak link exit"]):
                speak("Exiting assistant. Goodbye.")
                log("Exit command received.")
                break

            # SEARCH
            if "search" in cmd:
                speak("Please say the word to search for.")
                word = listen(timeout=8, phrase_time_limit=4)
                if not word:
                    speak("No word heard. Cancelling search.")
                    continue
                pages = search_word_in_pdf(filepath, word)
                if pages:
                    speak(f"Word {word} found on pages {', '.join(map(str,pages))}.")
                    log(f"Search results: {pages}")
                else:
                    speak(f"Word {word} not found.")
                    log("Search found nothing.")
                continue

            # QUERY (Q&A)
            if "query" in cmd:
                speak("Please ask your question now.")
                q = listen(timeout=12, phrase_time_limit=8)
                if not q:
                    speak("No question heard. Cancelling.")
                    continue
                speak("Finding an answer.")
                text = extract_text_from_pdf(filepath)
                ans = answer_query_t5(q, text) if TRANSFORMERS_AVAILABLE else "Q&A model not available."
                speak(ans)
                log("Spoken answer to query.")
                continue

            # SUMMARIZE
            if "summarize" in cmd or "summarise" in cmd:
                speak("Summarizing the document now.")
                summ = llm_summarize(filepath) if TRANSFORMERS_AVAILABLE else "Summarization model not available."
                # speak the summary in chunks
                for chunk in re.split(r'(?<=[.?!])\s+', summ):
                    speak(chunk)
                log("Summarization complete.")
                continue

            # READ ALOUD (entire doc)
            if "read aloud" in cmd:
                speak("Reading the document. Say stop to interrupt.")
                for p_idx in range(len(pdf_reader.pages)):
                    # read page p_idx
                    read_page(pdf_reader, p_idx, recognizer)
                    # quick check for stop
                    q = listen(timeout=0.6, phrase_time_limit=1)
                    if q and "stop" in q:
                        speak("Stopped reading.")
                        break
                continue

            # START NAVIGATION
            if "start navigation" in cmd or "navigation" in cmd:
                speak("Navigation started. Say next page, previous page, or page N. Say stop navigation to quit navigation.")
                current_page = 0
                while True:
                    nav = listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT)
                    if not nav:
                        speak("No navigation command heard; listening again.")
                        continue
                    log(f"Navigation command: {nav}")
                    if "next" in nav:
                        current_page = min(current_page + 1, total_pages - 1)
                    elif "previous" in nav:
                        current_page = max(current_page - 1, 0)
                    elif "page" in nav:
                        pn = parse_page_number_from_command(nav)
                        if pn and 1 <= pn <= total_pages:
                            current_page = pn - 1
                        else:
                            speak("Invalid page number.")
                            continue
                    elif any(w in nav for w in ["stop navigation", "stop", "exit navigation"]):
                        speak("Stopping navigation mode.")
                        break
                    else:
                        speak("Navigation command not recognized.")
                        continue
                    speak(f"Reading page {current_page + 1}.")
                    read_page(pdf_reader, current_page, recognizer)
                continue

            # HELP
            if "help" in cmd:
                speak("Available commands: search, query, summarize, read aloud, start navigation, help, and exit.")
                continue

            # fallback
            speak("Command not recognized. Say help for available commands.")
            continue

    except Exception as e:
        log(f"[ASSISTANT ERROR] {e}")
        traceback.print_exc()
    finally:
        log("Assistant loop terminated.")

# ---------- Streamlit UI & thread management ----------
st.set_page_config(layout="wide")
st.title("SPEAKLINK — Voice Automated PDF Assistant (Streamlit)")

st.write("This app runs the voice assistant in the background (server-side audio). The server must have microphone & speakers.")

# small instructions
with st.expander("How to use (important)"):
    st.markdown(
        """
- Add PDFs into the `data/` folder (create it if needed).
- Run `streamlit run app_streamlit.py` on the machine that has microphone & speakers.
- The assistant will ask for a filename by voice, then accept voice commands (search, query, summarize, read aloud, navigation, help, exit).
"""
    )

# start background assistant once per session
if "assistant_started" not in st.session_state:
    st.session_state["assistant_started"] = True
    threading.Thread(target=assistant_loop, daemon=True).start()
    log("Assistant thread launched.")

# show live logs
log_box = st.empty()
while True:
    new_lines = drain_logs()
    if new_lines:
        # append to existing shown text
        prev = log_box.text_area("Assistant log", height=300, value="", key="__log_display__", placeholder="")  # placeholder will be ignored after first run
        # Instead of trying to preserve previous state (Streamlit rerun limitations), show last N lines
        display_lines = []
        # gather current content from queue variable by recombining (we already drained)
        # We will collect lines from stored session_state for persistence
        if "log_lines" not in st.session_state:
            st.session_state["log_lines"] = []
        st.session_state["log_lines"].extend(new_lines)
        # limit to last 500 lines
        st.session_state["log_lines"] = st.session_state["log_lines"][-500:]
        log_box.text_area("Assistant log", value="\n".join(st.session_state["log_lines"]), height=300)
    time.sleep(0.8)
