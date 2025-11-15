# app_streamlit_voice_load.py
"""
Streamlit app — voice load PDF from ./data/ by filename (no manual click required).
Run:
    streamlit run app_streamlit_voice_load.py

Server must have a microphone + speakers for the Python process.
"""
import os
import re
import time
import threading
import queue
import base64
import traceback

import streamlit as st
import PyPDF2
import pyttsx3
import speech_recognition as sr

# ---------- Config ----------
DATA_DIR = "data"
ASR_TIMEOUT = 8
ASR_PHRASE_LIMIT = 6
LOG_Q = queue.Queue(maxsize=500)
SELECTED_FILE = None
SELECTED_FILE_LOCK = threading.Lock()

# ---------- Logging helper ----------
def log(msg: str):
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

# ---------- TTS setup ----------
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
            # non-blocking
            threading.Thread(target=tts_engine.runAndWait, daemon=True).start()
    except Exception as e:
        log(f"[TTS ERROR] {e}")

# ---------- ASR setup ----------
recognizer = sr.Recognizer()

def listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT):
    """Listen from server microphone and return lowercase text, or None."""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            if isinstance(text, str):
                return text.lower().strip()
            return None
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

# ---------- PDF utilities ----------
def list_data_pdfs():
    os.makedirs(DATA_DIR, exist_ok=True)
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

def display_pdf_base64(filepath):
    """Return HTML iframe embedding for a PDF file (base64)."""
    try:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700" type="application/pdf"></iframe>'
        return html
    except Exception as e:
        log(f"[PDF preview error] {e}")
        return None

def extract_text_all(filepath):
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

# ---------- Voice filename loader (background thread) ----------
def voice_filename_loader():
    """
    Runs in background thread. Asks user to speak filename and sets SELECTED_FILE global.
    """
    global SELECTED_FILE
    try:
        speak("Hello. I will listen for the PDF filename to open from the data folder.")
        time.sleep(0.2)

        pdfs = list_data_pdfs()
        if pdfs:
            speak(f"I found {len(pdfs)} PDF files in the data folder. They are:")
            for fn in pdfs:
                # speak filename - try not to block too long
                speak(fn.replace(".pdf", "").replace("_", " ").replace("-", " "), block=False)
                time.sleep(0.05)
        else:
            speak("No PDF files present in the data folder. Please add a file named sample.pdf to the data folder and try again.")

        # Ask for filename (max attempts)
        attempts = 0
        chosen = None
        while attempts < 6 and chosen is None:
            attempts += 1
            speak("Please say the filename you want to open, without saying dot PDF. Say quit to exit.")
            heard = listen(timeout=8, phrase_time_limit=5)
            if not heard:
                speak("I did not catch that. Please repeat the filename.")
                log("No filename heard; retrying.")
                continue
            log(f"Filename heard: {heard}")

            if any(word in heard for word in ["quit", "exit", "stop"]):
                speak("Exiting filename selection.")
                log("User opted to exit during filename step.")
                return

            # tolerant candidate variants
            candidate = heard.replace("underscore", "_").replace("dash", "-").replace("space", " ").strip()
            # multiple variants to match typical naming
            variants = [
                os.path.join(DATA_DIR, candidate + ".pdf"),
                os.path.join(DATA_DIR, candidate.replace(" ", "") + ".pdf"),
                os.path.join(DATA_DIR, candidate.replace(" ", "_") + ".pdf"),
                os.path.join(DATA_DIR, candidate.replace(" ", "-") + ".pdf"),
            ]
            for v in variants:
                if os.path.exists(v):
                    chosen = v
                    break

            if chosen:
                with SELECTED_FILE_LOCK:
                    SELECTED_FILE = chosen
                speak(f"Found file {os.path.basename(chosen)}. Loading it now.")
                log(f"Selected file: {chosen}")
                return
            else:
                speak(f"Could not find a file matching {candidate}. Please try again.")
                log(f"No file matched for: {candidate}")
                # after a couple attempts list options
                if attempts == 2 and pdfs:
                    speak("Available files are:")
                    for fn in pdfs:
                        speak(fn.replace(".pdf", "").replace("_", " "), block=False)
                        time.sleep(0.05)
                continue

        # If we exit loop with no selection
        speak("No valid filename selected. Please add a file to the data folder and reload the app.")
        log("Filename selection failed after attempts.")
    except Exception as e:
        log(f"[voice_filename_loader error] {e}")
        traceback.print_exc()

# ---------- Assistant main loop (background) ----------
def assistant_main_loop():
    """
    Waits until SELECTED_FILE is set, then loads file and continues voice-driven commands.
    This thread handles operations after file selection: search, read, summarize, navigation, help, exit.
    """
    global SELECTED_FILE
    # Wait for a selected file (set by voice_filename_loader)
    timeout_seconds = 300  # give the user time to respond
    waited = 0
    while True:
        with SELECTED_FILE_LOCK:
            local = SELECTED_FILE
        if local:
            break
        time.sleep(0.5)
        waited += 0.5
        if waited > timeout_seconds:
            log("Timeout waiting for filename selection.")
            speak("Timed out waiting for filename. Restart the app to try again.")
            return

    filepath = local
    # open PDF
    try:
        reader = PyPDF2.PdfReader(open(filepath, "rb"))
        total_pages = len(reader.pages)
        log(f"Opened {filepath} with {total_pages} pages.")
        speak(f"File {os.path.basename(filepath)} loaded. Document has {total_pages} pages.")
    except Exception as e:
        log(f"[PDF open error] {e}")
        speak("Failed to open the selected PDF.")
        return

    # Display will be handled by UI polling SELECTED_FILE (see UI code below)

    # Main voice-driven loop after file loaded
    while True:
        speak("Listening for a command.")
        cmd = listen(timeout=8, phrase_time_limit=6)
        if not cmd:
            log("No command heard; retrying.")
            time.sleep(0.3)
            continue
        log(f"Command heard: {cmd}")

        # EXIT
        if any(x in cmd for x in ["exit", "quit", "speaklink exit", "speak link exit"]):
            speak("Exiting assistant. Goodbye.")
            log("Exit command received.")
            break

        # HELP
        if "help" in cmd:
            speak("Available commands: search, read page N, read aloud, start navigation, summarize, query, help, exit.")
            continue

        # SEARCH
        if "search" in cmd:
            speak("Please say the search word.")
            word = listen(timeout=6, phrase_time_limit=4)
            if not word:
                speak("No word heard. Cancelling search.")
                continue
            log(f"Searching for: {word}")
            # perform search
            pages_found = []
            try:
                for i, p in enumerate(reader.pages):
                    ptext = p.extract_text() or ""
                    if word.lower() in ptext.lower():
                        pages_found.append(i + 1)
            except Exception as e:
                log(f"[search error] {e}")
            if pages_found:
                s = ", ".join(map(str, pages_found))
                speak(f"Word {word} found on page(s): {s}.")
                log(f"Search result pages: {s}")
            else:
                speak(f"Word {word} not found in the document.")
                log("Search found nothing.")
            continue

        # READ PAGE N
        if "read page" in cmd or ("read" in cmd and "page" in cmd):
            num = None
            nums = re.findall(r'\d+', cmd)
            if nums:
                try:
                    num = int(nums[0])
                except:
                    num = None
            if not num:
                speak("Which page number? Please say the page number.")
                resp = listen(timeout=6, phrase_time_limit=3)
                if resp:
                    nums = re.findall(r'\d+', resp)
                    if nums:
                        num = int(nums[0])
            if num and 1 <= num <= total_pages:
                speak(f"Reading page {num}.")
                try:
                    page_text = reader.pages[num-1].extract_text() or ""
                    # read sentences
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', page_text) if s.strip()]
                    for s in sentences:
                        speak(s)
                        # quick stop check
                        quick = listen(timeout=0.6, phrase_time_limit=1)
                        if quick and "stop" in quick:
                            speak("Stopped reading page.")
                            break
                except Exception as e:
                    log(f"[read page error] {e}")
                    speak("Could not read that page.")
            else:
                speak("Invalid page number.")
            continue

        # READ ALOUD (entire document)
        if "read aloud" in cmd or ("read" in cmd and "entire" in cmd):
            speak("Reading the entire document. Say stop to interrupt.")
            for idx in range(total_pages):
                try:
                    page_text = reader.pages[idx].extract_text() or ""
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', page_text) if s.strip()]
                    for s in sentences:
                        speak(s)
                        quick = listen(timeout=0.6, phrase_time_limit=1)
                        if quick and "stop" in quick:
                            speak("Stopped reading.")
                            break
                    if quick and "stop" in quick:
                        break
                except Exception as e:
                    log(f"[read aloud error] {e}")
            continue

        # START NAVIGATION
        if "start navigation" in cmd or "navigation" in cmd:
            speak("Starting navigation. Say next page, previous page, or page number N. Say stop navigation to exit.")
            cur = 0
            while True:
                nav = listen(timeout=8, phrase_time_limit=5)
                if not nav:
                    speak("No navigation command heard. Listening again.")
                    continue
                log(f"Navigation command: {nav}")
                if "next" in nav:
                    cur = min(cur + 1, total_pages - 1)
                elif "previous" in nav:
                    cur = max(cur - 1, 0)
                elif "page" in nav:
                    n = re.findall(r'\d+', nav)
                    if n:
                        p = int(n[0])
                        if 1 <= p <= total_pages:
                            cur = p - 1
                        else:
                            speak("Invalid page number.")
                            continue
                    else:
                        speak("Page number not detected.")
                        continue
                elif any(w in nav for w in ["stop navigation", "stop", "exit navigation"]):
                    speak("Stopping navigation.")
                    break
                else:
                    speak("Navigation command not recognized.")
                    continue

                speak(f"Reading page {cur + 1}.")
                try:
                    page_text = reader.pages[cur].extract_text() or ""
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', page_text) if s.strip()]
                    for s in sentences:
                        speak(s)
                        quick = listen(timeout=0.6, phrase_time_limit=1)
                        if quick and "stop" in quick:
                            speak("Stopped reading.")
                            break
                except Exception as e:
                    log(f"[nav read error] {e}")
            continue

        # SUMMARIZE or QUERY are omitted here to keep the example focused on voice-load; can be added.
        if "summarize" in cmd or "summarise" in cmd:
            speak("Summarization not configured in this build. If you want LLM summarization, ask me to add it.")
            continue

        if "query" in cmd:
            speak("Query/Q&A not configured in this build. Ask me to enable LLM features if needed.")
            continue

        # fallback
        speak("Command not recognized. Say help to get available commands.")
        continue

# ---------- Start background threads ----------
if "voice_thread_started" not in st.session_state:
    st.session_state["voice_thread_started"] = True
    # thread to get filename via voice
    threading.Thread(target=voice_filename_loader, daemon=True).start()
    # assistant main loop thread (waits for selection)
    threading.Thread(target=assistant_main_loop, daemon=True).start()
    log("Background voice threads started.")

# ---------- Streamlit UI (polls for selected file & logs) ----------
st.set_page_config(layout="wide")
st.title("SpeakLink — Voice-based PDF Loader (Streamlit)")

st.markdown(
    "This app will **listen** for a PDF filename (speak the filename without `.pdf`) and load it from the `data/` folder. "
    "The server must have a microphone & speakers for audio I/O."
)

# show selected file (polled)
with st.expander("Current status / loaded file", expanded=True):
    with st.spinner("Waiting for voice filename selection..."):
        # poll for selected file for a short period to update UI after thread sets it
        displayed = None
        for _ in range(10):
            with SELECTED_FILE_LOCK:
                displayed = SELECTED_FILE
            if displayed:
                break
            time.sleep(0.4)

        if displayed:
            st.success(f"Loaded file: {os.path.basename(displayed)}")
            html = display_pdf_base64(displayed)
            if html:
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning("Could not display PDF preview.")
        else:
            st.info("No file loaded yet. Speak the filename — watch the assistant speak its prompts (server audio).")

# live log area
log_box = st.empty()
if "log_lines" not in st.session_state:
    st.session_state["log_lines"] = []

# small loop to show logs live (Streamlit will re-run this script periodically)
new = drain_logs()
if new:
    st.session_state["log_lines"].extend(new)
    st.session_state["log_lines"] = st.session_state["log_lines"][-500:]

log_box.text_area("Assistant log (latest first at top)", value="\n".join(reversed(st.session_state["log_lines"])), height=300)

st.caption("If the app can't hear you, check microphone permissions and ensure Streamlit is running on a machine with mic access.")
