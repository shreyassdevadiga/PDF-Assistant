# app.py
import os
import re
import base64
import time
import streamlit as st
import streamlit.components.v1 as components

import PyPDF2
import pyttsx3
import speech_recognition as sr

# Optional transformers imports (lazy loaded)
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ========== CONFIG ==========
MODEL_CHECKPOINT = "MBZUAI/LaMini-Flan-T5-248M"  # keep as-is; if offline, comment out summarization parts
# ========== /CONFIG ==========

# Initialize TTS engine once
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    return engine

TTS_ENGINE = init_tts()

# Small wrapper so code uses same speak() consistently
def speak(text: str):
    if text is None or text == "":
        return
    try:
        TTS_ENGINE.say(str(text))
        TTS_ENGINE.runAndWait()
    except Exception as e:
        # TTS should not crash app - log to Streamlit
        st.sidebar.error(f"TTS error: {e}")

# -----------------------------
# PDF helpers
# -----------------------------
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def search_word_in_pdf(filepath, search_word):
    try:
        pdf_reader = PyPDF2.PdfReader(filepath)
    except Exception as e:
        st.sidebar.error(f"Failed to open PDF: {e}")
        return []
    total_pages = len(pdf_reader.pages)
    occurrences = []
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text() or ""
        if search_word.lower() in page_text.lower():
            occurrences.append(page_num + 1)  # 1-based page numbering
    return occurrences

def displayPDF(file):
    # Try to use pdf viewer package first, otherwise embed base64
    try:
        from streamlit_pdf_viewer import pdf_viewer
        with open(file, "rb") as f:
            pdf_bytes = f.read()
        pdf_viewer(pdf_bytes, width=1000, height=700)
        return
    except Exception:
        pass

    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        blob_html = f"""
        <div id="pdf_container" style="height:700px;">
          <iframe id="pdf_iframe" style="width:100%;height:700px;border:none;"></iframe>
        </div>
        <script>
        (function() {{
          const b64 = "{base64_pdf}";
          function b64ToUint8Array(b64) {{
            const binary = atob(b64);
            const len = binary.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
            return bytes;
          }}
          const bytes = b64ToUint8Array(b64);
          const blob = new Blob([bytes], {{ type: 'application/pdf' }});
          const url = URL.createObjectURL(blob);
          const iframe = document.getElementById('pdf_iframe');
          iframe.src = url;
        }})();
        </script>
        """
        components.html(blob_html, height=700, scrolling=False)
    except Exception as e:
        st.warning("Unable to preview PDF. Please check the file path or try again.")
        st.sidebar.error(f"Preview error: {e}")

# -----------------------------
# Language model helpers (lazy load)
# -----------------------------
LLM_PIPELINE = None
TOKENIZER = None
BASE_MODEL = None

def load_llm_models():
    global LLM_PIPELINE, TOKENIZER, BASE_MODEL
    if LLM_PIPELINE is not None:
        return
    try:
        # load tokenizer & model (may take time)
        TOKENIZER = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
        BASE_MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT, torch_dtype=None)
        # summarization pipeline using model/tokenizer
        LLM_PIPELINE = pipeline(
            "summarization",
            model=BASE_MODEL,
            tokenizer=TOKENIZER,
            device=-1  # CPU; change if GPU available
        )
    except Exception as e:
        st.sidebar.warning(f"Could not load LLM models: {e}")
        LLM_PIPELINE = None

def llm_summarize(filepath):
    load_llm_models()
    if LLM_PIPELINE is None:
        return "Summarization model is not available. Install/model download failed."
    text = file_preprocessing(filepath)
    # pipeline may expect chunks; for safety, truncate
    if len(text) > 30000:
        text = text[:30000]
    try:
        out = LLM_PIPELINE(text, max_length=300, min_length=30)
        return out[0].get("summary_text", "")
    except Exception as e:
        st.sidebar.error(f"Summarization error: {e}")
        return "Summarization failed."

def file_preprocessing(file):
    try:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for t in texts:
            final_texts += t.page_content + "\n"
        return final_texts
    except Exception as e:
        # fallback to raw extraction
        return extract_text_from_pdf(file)

def answer_query_t5(query, text):
    # Lazy load tokenizer and model if not already
    try:
        global TOKENIZER, BASE_MODEL
        if TOKENIZER is None or BASE_MODEL is None:
            TOKENIZER = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
            BASE_MODEL = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
    except Exception as e:
        st.sidebar.error(f"LLM load error: {e}")
        return "Model not available."

    prompt = f"question: {query} context: {text}"
    try:
        inputs = TOKENIZER.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        answer_ids = BASE_MODEL.generate(inputs, max_length=200, num_return_sequences=1, num_beams=3, early_stopping=True)
        answer = TOKENIZER.decode(answer_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        st.sidebar.error(f"Error generating answer: {e}")
        return "Failed to generate answer."

# -----------------------------
# Speech listening helpers
# -----------------------------
def extract_page_number(command: str):
    # tries to find a number after the word "page"
    try:
        parts = command.split()
        if "page" in parts:
            idx = parts.index("page")
            if idx + 1 < len(parts):
                # attempt integer parse
                try:
                    return int(parts[idx + 1])
                except ValueError:
                    # fallback: no numeric found
                    return None
    except Exception:
        return None

def listen_for_command(recognizer: sr.Recognizer, timeout=8, phrase_limit=5):
    """Listens and returns lowercase string, or None on failure/timeouts."""
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            text = recognizer.recognize_google(audio)
            return text.lower()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            st.sidebar.error(f"Speech recognition request failed: {e}")
            return None
        except Exception as e:
            st.sidebar.error(f"Microphone/listen error: {e}")
            return None

def listen_for_question(recognizer):
    speak("Listening for your question.")
    return listen_for_command(recognizer, timeout=10, phrase_limit=8)

def listen_for_search_word(recognizer):
    speak("Please speak the word to search for.")
    return listen_for_command(recognizer, timeout=8, phrase_limit=4)

# -----------------------------
# Reader helpers
# -----------------------------
def read_page(pdf_reader, page_number, speaker_engine=None, recognizer=None):
    # page_number: 0-based index
    if page_number < 0 or page_number >= len(pdf_reader.pages):
        st.sidebar.warning("Invalid page number to read.")
        return
    page = pdf_reader.pages[page_number]
    text = page.extract_text() or ""

    # split into sentences using simple dot rule
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
    for sentence in sentences:
        speak(sentence)
        # check for immediate stop brief command
        if recognizer:
            cmd = listen_for_command(recognizer, timeout=0.8, phrase_limit=2)
            if cmd and "stop" in cmd:
                speak("Stopped reading.")
                return

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(layout="wide")
def main():
    st.title("SPEAKLINK — Voice Automated PDF Assistant")

    recognizer = sr.Recognizer()

    # greet once
    greeting_message = (
        "Hello, I am SpeakLink, a virtual assistant at your service. "
        "You can give me commands as soon as I say 'listening for a command.' "
        "My capabilities include reading, summarizing, navigating, querying and searching. "
        "Say 'help' for available commands and 'speaklink exit' to exit."
    )
    st.sidebar.info(greeting_message)
    speak(greeting_message)

    # Choose file either by voice (existing in /data) OR by file uploader
    st.sidebar.header("Open PDF")
    upload_choice = st.sidebar.radio("Open using", ["Upload file", "Voice filename (from data/ folder)"])

    filepath = None
    if upload_choice == "Upload file":
        uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded:
            save_path = os.path.join("data", uploaded.name)
            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            filepath = save_path
            st.sidebar.success(f"Saved to {save_path}")
    else:
        # voice filename mode
        st.sidebar.info("Say the filename (without .pdf) from the data/ folder, or click the 'Listen for filename' button.")
        if st.sidebar.button("Listen for filename"):
            speak("Listening for the file name.")
            name = listen_for_command(recognizer, timeout=8, phrase_limit=5)
            if not name:
                st.sidebar.error("No filename heard. Try again or upload file.")
            else:
                # tolerate 'underscore' and 'space' words
                file_name = name.replace("underscore", "_").replace(" dash ", "-").replace(" space ", " ").strip()
                file_name = file_name.replace(" ", "")
                candidate = os.path.join("data", f"{file_name}.pdf")
                if os.path.exists(candidate):
                    filepath = candidate
                    st.sidebar.success(f"Found file: {candidate}")
                    speak("File located successfully.")
                else:
                    st.sidebar.error(f"File {candidate} not found. Please upload or try again.")

    # If filepath not found yet, show sample data files if any
    if not filepath:
        data_files = []
        if os.path.isdir("data"):
            for fn in os.listdir("data"):
                if fn.lower().endswith(".pdf"):
                    data_files.append(os.path.join("data", fn))
        if data_files:
            selected = st.sidebar.selectbox("Or pick a file from data/", [""] + data_files)
            if selected:
                filepath = selected
        else:
            st.sidebar.info("No local PDFs found in data/. Upload one to get started.")

    if not filepath:
        st.info("Waiting for PDF. Upload or select a file from the sidebar to start.")
        return

    # open pdf for operations
    try:
        pdf_reader = PyPDF2.PdfReader(filepath)
        total_pages = len(pdf_reader.pages)
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        return

    current_page_index = 0

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Document Preview")
        displayPDF(filepath)

    with col2:
        st.header("Voice Console")
        st.write("Press 'Listen for command' and then speak one of the available commands.")
        if st.button("Listen for command"):
            speak("Listening for a command now.")
            command = listen_for_command(recognizer)
            if not command:
                st.sidebar.warning("No command detected. Try again.")
            else:
                st.sidebar.info(f"Command: {command}")

                # SEARCH
                if "search" in command:
                    word = listen_for_search_word(recognizer)
                    if not word:
                        st.sidebar.warning("No search word heard.")
                        speak("No word heard.")
                    else:
                        pages = search_word_in_pdf(filepath, word)
                        if pages:
                            speak(f"Word {word} found on pages {', '.join(map(str, pages))}.")
                            st.success(f"Found on pages: {pages}")
                        else:
                            speak(f"Word {word} not found in the document.")
                            st.warning("Not found.")

                # QUERY (LLM Q&A)
                elif "query" in command:
                    speak("Please ask your question now.")
                    question = listen_for_question(recognizer)
                    if not question:
                        st.sidebar.warning("No question heard.")
                    else:
                        txt = extract_text_from_pdf(filepath)
                        answer = answer_query_t5(question, txt)
                        st.success(f"Answer: {answer}")
                        speak(answer)

                # SUMMARIZE
                elif "summarize" in command or "summarise" in command:
                    st.sidebar.info("Summarizing (this may take a while if model needs downloading).")
                    speak("Summarizing the document now.")
                    summary = llm_summarize(filepath)
                    st.success("Summary:")
                    st.write(summary)
                    speak("Here is the summary.")
                    speak(summary)

                # READ ALOUD (entire doc)
                elif "read aloud" in command or "read entire" in command:
                    speak("Reading the entire document. Say 'stop' to interrupt.")
                    for page_idx in range(len(pdf_reader.pages)):
                        read_page(pdf_reader, page_idx, recognizer=recognizer)
                        # quick check to break if user says stop
                        cmd = listen_for_command(recognizer, timeout=0.4, phrase_limit=2)
                        if cmd and "stop" in cmd:
                            speak("Stopped reading.")
                            break

                # NAVIGATION
                elif "start navigation" in command or "navigation" in command:
                    speak("Starting navigation. Say 'next page', 'previous page', or 'page X'. Say 'stop navigation' to exit.")
                    while True:
                        nav_cmd = listen_for_command(recognizer, timeout=8, phrase_limit=4)
                        if not nav_cmd:
                            st.sidebar.info("No nav command detected. Try again or say 'stop navigation'.")
                            continue
                        if "next" in nav_cmd:
                            current_page_index = min(current_page_index + 1, total_pages - 1)
                        elif "previous" in nav_cmd:
                            current_page_index = max(current_page_index - 1, 0)
                        elif "page" in nav_cmd:
                            p = extract_page_number(nav_cmd)
                            if p and 1 <= p <= total_pages:
                                current_page_index = p - 1
                            else:
                                speak("Invalid page number.")
                                continue
                        elif "stop navigation" in nav_cmd or "stop" in nav_cmd:
                            speak("Stopping navigation.")
                            break
                        else:
                            speak("Navigation command not recognized.")
                            continue

                        # read selected page
                        speak(f"Reading page {current_page_index + 1}.")
                        read_page(pdf_reader, current_page_index, recognizer=recognizer)

                # HELP
                elif "help" in command:
                    help_text = (
                        "Available voice commands: search, query, summarize, read aloud, start navigation, help, speaklink exit."
                    )
                    st.sidebar.info(help_text)
                    speak(help_text)

                # EXIT
                elif any(phrase in command for phrase in ["exit", "quit", "speaklink exit", "speak link exit"]):
                    speak("Exiting the assistant. Goodbye.")
                    st.sidebar.info("Exiting assistant.")
                    return

                else:
                    st.sidebar.warning("Command not recognized. Say help for available commands.")
                    speak("Command not recognized. Say help for available commands.")

        # Show simple manual controls as well
        st.write("---")
        st.subheader("Manual controls (non-voice)")
        st.write(f"Total pages: {total_pages}")
        page_selected = st.number_input("Go to page (1-based)", min_value=1, max_value=total_pages, value=1, step=1)
        if st.button("Read selected page"):
            read_page(pdf_reader, page_selected - 1, recognizer=recognizer)
        if st.button("Summarize (start)"):
            summary = llm_summarize(filepath)
            st.write(summary)
            speak("Summary ready, shown on screen.")

if __name__ == "__main__":
    main()
