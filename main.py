# app_streamlit.py
"""
Streamlit wrapper for the fully voice automated PDF assistant.

Run:
    streamlit run app_streamlit.py

Important:
- This app runs the voice assistant in a background thread on the server.
- The server must have a working microphone and speakers available to the Python process.
- The assistant is voice-driven only (no manual interaction required).
"""

import os
import re
import time
import threading
import queue
import traceback

import streamlit as st
import PyPDF2
import pyttsx3
import speech_recognition as sr

# Optional transformers (only used if installed & desired)
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -------------- Configuration --------------
DATA_DIR = "data"
MODEL_CHECKPOINT = "MBZUAI/LaMini-Flan-T5-248M"
ASR_TIMEOUT = 8
ASR_PHRASE_LIMIT = 6
LISTEN_RETRY_DELAY = 0.6
# -------------------------------------------

# UI: a queue for log messages from background thread
LOG_Q = queue.Queue(maxsize=200)

def log(msg: str):
    """Add a message to the UI log (thread-safe)."""
    try:
        LOG_Q.put_nowait(f"{time.strftime('%H:%M:%S')} — {msg}")
    except queue.Full:
        pass

# Initialize TTS engine once
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

# Initialize recognizer
recognizer = sr.Recognizer()

def listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT):
    """Listen with server microphone and return recognized text or None."""
    with sr.Microphone() as source:
        try:
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

# ---------- PDF utilities (same behavior as previous assistant) ----------
def list_data_pdfs():
    os.makedirs(DATA_DIR, exist_ok=True)
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

def load_pdf_reader(filepath):
    try:
        f = open(filepath, "rb")
        pdf_reader = PyPDF2.PdfReader(f)
        return pdf_reader, f
    except Exception as e:
        log(f"[PDF LOAD ERROR] {e}")
        return None, None

def extract_text_from_pdf_reader(pdf_reader):
    texts = []
    try:
        for p in pdf_reader.pages:
            texts.append(p.extract_text() or "")
    except Exception as e:
        log(f"[PDF EXTRACT ERROR] {e}")
    return texts

def search_word_in_pages(pages_texts, word):
    results = []
    if not word:
        return results
    w = word.lower()
    for i, ptext in enumerate(pages_texts):
        if w in (ptext or "").lower():
            results.append(i + 1)
    return results

def read_page_text(pages_texts, page_idx, stop_event):
    if page_idx < 0 or page_idx >= len(pages_texts):
        speak("Invalid page number.")
        return
    text = pages_texts[page_idx] or ""
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    for sent in sentences:
        if stop_event.is_set():
            speak("Stopped reading.")
            return
        speak(sent)
        time.sleep(0.05)

# ---------- LLM helpers (optional) ----------
llm_pipeline = None
tokenizer = None
base_model = None
llm_loaded = False

def try_load_llm():
    global llm_pipeline, tokenizer, base_model, llm_loaded
    if llm_loaded:
        return True
    if not TRANSFORMERS_AVAILABLE:
        log("Transformers not installed; skipping LLM load.")
        return False
    try:
        log("Loading LLM model (may take time)...")
        tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
        base_model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
        llm_pipeline = pipeline('summarization', model=base_model, tokenizer=tokenizer)
        llm_loaded = True
        log("LLM loaded.")
        return True
    except Exception as e:
        log(f"[LLM LOAD ERROR] {e}")
        llm_loaded = False
        return False

def summarize_text_concat(pages_texts, max_chars=30000):
    if not try_load_llm():
        return "Summarization model unavailable."
    full = "\n".join(pages_texts)
    if len(full) > max_chars:
        full = full[:max_chars]
    try:
        out = llm_pipeline(full, max_length=300, min_length=30)
        return out[0].get("summary_text", "")
    except Exception as e:
        log(f"[LLM SUMMARIZE ERROR] {e}")
        return "Summarization failed."

def answer_query_t5(query, pages_texts):
    if not try_load_llm():
        return "Q&A model unavailable."
    full = "\n".join(pages_texts)
    prompt = f"question: {query} context: {full}"
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        answer_ids = base_model.generate(inputs, max_length=200, num_return_sequences=1, num_beams=3, early_stopping=True)
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        log(f"[LLM QA ERROR] {e}")
        return "Failed to generate answer."

# ---------- Command parsing & loop ----------
def parse_page_number_from_command(command):
    try:
        words = re.findall(r'\d+', command)
        if words:
            return int(words[0])
    except Exception:
        pass
    return None

def announce_available_commands():
    msg = ("Available commands: search, query, summarize, read aloud, read page, start navigation, next page, previous page, "
           "page number N, help, and exit.")
    speak(msg)
    log("Announced available commands.")

def run_voice_loop(pdf_reader, pages_texts):
    total_pages = len(pages_texts)
    current_page_idx = 0
    stop_event = threading.Event()

    speak(f"Document loaded. It has {total_pages} pages.")
    log(f"Document loaded with {total_pages} pages.")
    time.sleep(0.3)
    announce_available_commands()

    while True:
        speak("Listening for command.")
        cmd = listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT)
        if not cmd:
            log("No command detected; retrying.")
            time.sleep(LISTEN_RETRY_DELAY)
            continue

        log(f"Heard command: {cmd}")
        print(f"[CMD] {cmd}")

        if any(phrase in cmd for phrase in ["exit", "quit", "speaklink exit", "speak link exit", "stop assistant"]):
            speak("Exiting. Goodbye.")
            log("Exit command received.")
            break

        if "help" in cmd:
            announce_available_commands()
            continue

        if "search" in cmd:
            speak("Please speak the search word.")
            word = listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT)
            if not word:
                speak("No search word heard.")
                log("Search cancelled: no word.")
                continue
            log(f"Searching for: {word}")
            pages = search_word_in_pages(pages_texts, word)
            if pages:
                s = ", ".join(map(str, pages))
                speak(f"Found {word} on pages {s}.")
                log(f"Search found on pages: {s}")
            else:
                speak(f"Word {word} not found.")
                log("Search found nothing.")
            continue

        if "query" in cmd:
            speak("Please ask your question now.")
            question = listen(timeout=12, phrase_time_limit=8)
            if not question:
                speak("No question heard.")
                log("Query cancelled: no question.")
                continue
            log(f"Question: {question}")
            answer = answer_query_t5(question, pages_texts)
            speak(answer)
            log(f"Answer spoken.")
            continue

        if "summarize" in cmd or "summarise" in cmd:
            speak("Summarizing now.")
            log("Summarization started.")
            summary = summarize_text_concat(pages_texts)
            # speak in chunks
            for chunk in re.split(r'(?<=[.?!])\s+', summary):
                speak(chunk)
            log("Summarization finished.")
            continue

        if "read aloud" in cmd or ("read" in cmd and "page" not in cmd and "navigation" not in cmd):
            speak("Reading the entire document now. Say stop to interrupt.")
            log("Reading entire document started.")
            stop_event.clear()
            for idx in range(len(pages_texts)):
                if stop_event.is_set():
                    break
                speak(f"Reading page {idx + 1}.")
                sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', pages_texts[idx] or "") if s.strip()]
                for s in sentences:
                    if stop_event.is_set():
                        break
                    speak(s)
                    quick = listen(timeout=0.6, phrase_time_limit=1)
                    if quick and "stop" in quick:
                        stop_event.set()
                        speak("Stopping reading.")
                        log("Read interrupted by stop command.")
                        break
            log("Completed read aloud (or interrupted).")
            continue

        if "read page" in cmd or (("read" in cmd) and ("page" in cmd)):
            pnum = parse_page_number_from_command(cmd)
            if pnum is None:
                speak("Which page number?")
                resp = listen()
                if resp:
                    pnum = parse_page_number_from_command(resp)
            if pnum and 1 <= pnum <= len(pages_texts):
                speak(f"Reading page {pnum}.")
                log(f"Reading page {pnum}.")
                stop_event.clear()
                read_page_text(pages_texts, pnum - 1, stop_event)
            else:
                speak("Invalid page number.")
                log("Invalid page read request.")
            continue

        if "navigation" in cmd or "start navigation" in cmd:
            speak("Starting navigation mode. Say next, previous, page N, or stop navigation.")
            log("Navigation mode started.")
            while True:
                nav_cmd = listen(timeout=ASR_TIMEOUT, phrase_time_limit=ASR_PHRASE_LIMIT)
                if not nav_cmd:
                    speak("No command detected in navigation; listening again.")
                    continue
                log(f"Navigation heard: {nav_cmd}")
                if "next" in nav_cmd:
                    current_page_idx = min(current_page_idx + 1, len(pages_texts) - 1)
                elif "previous" in nav_cmd:
                    current_page_idx = max(current_page_idx - 1, 0)
                elif "page" in nav_cmd:
                    pn = parse_page_number_from_command(nav_cmd)
                    if pn and 1 <= pn <= len(pages_texts):
                        current_page_idx = pn - 1
                    else:
                        speak("Invalid page. Say the page number again.")
                        continue
                elif any(w in nav_cmd for w in ["stop navigation", "stop", "exit navigation"]):
                    speak("Stopping navigation mode.")
                    log("Navigation mode stopped.")
                    break
                else:
                    speak("Navigation command not understood.")
                    continue

                speak(f"Now on page {current_page_idx + 1}. Reading it.")
                read_page_text(pages_texts, current_page_idx, threading.Event())
            continue

        # fallback
        speak("Command not recognized. Say help for the list of commands.")
        log("Unrecognized command.")
        continue

# ---------- Background thread entrypoint ----------
def assistant_background_thread():
    try:
        greeting = (
            "Hello, I am SpeakLink, your voice automated PDF assistant. "
            "I will listen for the filename in the data folder, then accept voice commands."
        )
        speak(greeting)
        log("Assistant started. Greeting spoken.")

        # list PDFs
        pdfs = list_data_pdfs()
        if pdfs:
            speak(f"I found {len(pdfs)} PDF files in the data folder.")
            for fn in pdfs:
                speak(fn.replace(".pdf", ""))
                time.sleep(0.05)
        else:
            speak("No PDFs found in the data folder. Add a PDF to the data folder and I will try to open it when you tell me its filename.")

        # ask for filename
        speak("Please say the filename you want to open, without the dot PDF extension. Say quit to exit.")
        tries = 0
        filename = None
        while tries < 6 and filename is None:
            tries += 1
            heard = listen(timeout=8, phrase_time_limit=5)
            if not heard:
                speak("I did not hear that, please say the filename again.")
                log("No filename heard; retrying.")
                continue
            log(f"Filename heard: {heard}")
            if any(x in heard for x in ["quit", "exit", "stop"]):
                speak("Exiting now. Goodbye.")
                log("Exit while waiting for filename.")
                return
            candidate_name = heard.replace("underscore", "_").replace("dash", "-").replace("space", " ").strip()
            variants = [
                os.path.join(DATA_DIR, candidate_name + ".pdf"),
                os.path.join(DATA_DIR, candidate_name.replace(" ", "") + ".pdf"),
                os.path.join(DATA_DIR, candidate_name.replace(" ", "_") + ".pdf"),
            ]
            chosen = None
            for c in variants:
                if os.path.exists(c):
                    chosen = c
                    break
            if chosen:
                filename = chosen
                speak(f"Found file {os.path.basename(chosen)}. Opening it now.")
                log(f"Selected file: {chosen}")
                break
            else:
                speak(f"Could not find file named {candidate_name} in data folder. Try again.")
                log(f"Filename not found: {candidate_name}")
                if tries == 2 and pdfs:
                    speak("Available files are:")
                    for fn in pdfs:
                        speak(fn.replace(".pdf", ""))
        if not filename:
            speak("No filename selected. Exiting.")
            log("No filename after retries. Assistant exiting.")
            return

        pdf_reader, fh = load_pdf_reader(filename)
        if pdf_reader is None:
            speak("Failed to open PDF.")
            log("Failed to open PDF file.")
            return
        pages_texts = extract_text_from_pdf_reader(pdf_reader)
        if not any(pages_texts):
            speak("Warning: PDF had no extractable text. Scanned PDFs need OCR which is not enabled.")
            log("PDF contains no extractable text.")

        # background load LLM (optional)
        if TRANSFORMERS_AVAILABLE:
            threading.Thread(target=try_load_llm, daemon=True).start()

        # start the command loop
        run_voice_loop(pdf_reader, pages_texts)

    except Exception as e:
        log(f"[ASSISTANT CRASH] {e}")
        traceback.print_exc()
    finally:
        log("Assistant background thread terminated.")

# ---------- Streamlit UI & thread management ----------
st.set_page_config(layout="wide")
st.title("SpeakLink — Voice Automated PDF Assistant (Streamlit)")
st.write("This page displays logs from the voice assistant. The assistant runs automatically in the background and is fully voice-driven (no manual clicks required).")
st.write("**Important:** The server running this app must have a microphone & speakers for the assistant to work.")

# show instructions
with st.expander("How it works (brief)"):
    st.markdown(
        """
- On first load the assistant will speak and ask for a filename from the `data/` folder.
- After opening the PDF it will accept voice commands: `search`, `query`, `summarize`, `read aloud`, `read page N`, `start navigation`, `help`, `exit`.
- All audio I/O is server-side: the process that runs Streamlit must have mic/speakers.
"""
    )

# show live log area
log_container = st.empty()

# Start the background thread only once per session
if "assistant_thread_started" not in st.session_state:
    st.session_state["assistant_thread_started"] = True
    t = threading.Thread(target=assistant_background_thread, daemon=True)
    t.start()
    log("Assistant thread launched.")

# Continuously pull log messages and display
def drain_logs():
    msgs = []
    while not LOG_Q.empty():
        try:
            msgs.append(LOG_Q.get_nowait())
        except queue.Empty:
            break
    return msgs

# simple loop to refresh logs periodically
def live_log_loop():
    all_msgs = []
    for _ in range(20):  # show recent messages; repeated Streamlit reruns will keep updating
        new = drain_logs()
        if new:
            all_msgs.extend(new)
        # display last 200 lines
        display = "\n".join(all_msgs[-200:])
        log_container.code(display, language="text")
        time.sleep(0.5)

# Run a short live update (Streamlit will rerun periodically; this keeps UI responsive)
live_log_loop()
