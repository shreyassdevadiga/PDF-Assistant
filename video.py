# app.py (corrected)
import os
import re
import base64
import streamlit as st
import PyPDF2
import pyttsx3
import speech_recognition as sr
from speech_recognition import WaitTimeoutError, UnknownValueError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import threading
#import
# -------------------------
# Configuration
# -------------------------
# Default fallback file (uploaded by you)
DEFAULT_LOCAL_PDF = "/mnt/data/Project synopsis.pdf"  # uploaded file path (used if voice selection fails)
# Change checkpoint if you have a different T5 model available
CHECKPOINT = "MBZUAI/LaMini-Flan-T5-248M"

# -------------------------
# Load models (may take time)
# -------------------------
@st.cache_data(show_spinner=False)
def load_models():
    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT, torch_dtype=torch.float32)
    # Create a summarization pipeline using model + tokenizer
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU; change to 0 for GPU if available
    )
    return tokenizer, model, summarizer

tokenizer, base_model, summarizer = load_models()

# -------------------------
# Helper functions
# -------------------------
def speak(text):
    def run():
        speaker.say(text)
        speaker.runAndWait()

    t = threading.Thread(target=run)
    t.start()

def display_pdf(filepath):
    """Embed and display a PDF in Streamlit page area."""
    if not os.path.exists(filepath):
        st.error(f"File not found: {filepath}")
        return
    with open(filepath, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def extract_text_from_pdf(filepath):
    """Extract all text from a PDF file (simple concatenation)."""
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            ptext = page.extract_text() or ""
            text += ptext + "\n"
    return text

def file_preprocessing(filepath):
    """Use PyPDFLoader + RecursiveCharacterTextSplitter (keeps same logic as before)."""
    loader = PyPDFLoader(filepath)
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # join chunk contents to pass into models that need a single string
    full_text = "\n".join([c.page_content for c in chunks])
    return full_text

def llm_summarize(filepath):
    """Summarize using the loaded summarizer pipeline."""
    text = file_preprocessing(filepath)
    # summarizer expects shorter inputs; if too long, we chunk and summarize progressively
    if len(text) > 4000:
        # naive chunk-based summarization
        parts = [text[i:i+3000] for i in range(0, len(text), 3000)]
        summaries = []
        for p in parts:
            out = summarizer(p, max_length=200, min_length=30, truncation=True)
            summaries.append(out[0]["summary_text"])
        # summarize the concatenated summaries
        joined = " ".join(summaries)
        final = summarizer(joined, max_length=250, min_length=30, truncation=True)[0]["summary_text"]
        return final
    else:
        return summarizer(text, max_length=300, min_length=50, truncation=True)[0]["summary_text"]

def answer_query_t5(query, text):
    """Use T5 model to answer question given context text."""
    prompt = f"question: {query} context: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    out_ids = base_model.generate(inputs, max_length=200, num_beams=3, early_stopping=True)
    answer = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return answer

def sentence_split(text):
    """Simple sentence splitter (robust enough for our use)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def search_word_in_pdf(filepath, search_word):
    """Return list of 1-based page numbers containing the search word."""
    reader = PyPDF2.PdfReader(filepath)
    found_pages = []
    for i, p in enumerate(reader.pages):
        text = (p.extract_text() or "").lower()
        if search_word.lower() in text:
            found_pages.append(i+1)
    return found_pages

def extract_page_number(command):
    """Extract 'page N' from a command, returns int or None."""
    try:
        # Look for word 'page' followed by a number
        m = re.search(r'\bpage\s+(\d{1,4})\b', command)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None

# -------------------------
# Robust listening utils
# -------------------------
def listen_once(recognizer, timeout=6, phrase_time_limit=8):
    """Listen once and return recognized text or None."""
    with sr.Microphone() as source:
        try:
            # adjust for ambient noise briefly
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            return text.lower()
        except WaitTimeoutError:
            return None
        except UnknownValueError:
            return None
        except sr.RequestError:
            return None

# -------------------------
# Main Streamlit app
# -------------------------
st.set_page_config(layout="wide")
st.title("SPEAKLINK — Voice Automated PDF Assistant for the Visually Impaired")

def main():
    # initialize
    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()

    # sidebar controls
    st.sidebar.header("Controls")
    uploaded = st.sidebar.file_uploader("Upload a PDF (optional). If you upload, voice filename selection will fall back to this.", type=["pdf"])
    use_default = False

    # Greet user once
    greeting = (
        "Hello, I am SpeakLink. I provide voice-based reading, search, navigation, summarization, and Q&A on PDFs. "
        "Say 'help' at any time to hear available commands. Say 'exit' to stop."
    )
    st.sidebar.info(greeting)
    speak(greeting, speaker)

    # Determine filepath: prefer uploader, then voice-based filename, then default local file
    filepath = None
    if uploaded is not None:
        # write uploaded to a temp file so other libs can read it
        temp_path = os.path.join("data", uploaded.name)
        os.makedirs("data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        filepath = temp_path
        st.sidebar.success(f"Uploaded {uploaded.name}")
    else:
        # Try voice-based filename selection (single attempt; fallback to default file)
        st.sidebar.info("Please say the file name now, or say 'use default' to open the project synopsis.")
        speak("Please say the file name now, or say 'use default' to open the project synopsis.", speaker)
        file_choice = listen_once(recognizer, timeout=6, phrase_time_limit=5)
        if file_choice is None:
            st.sidebar.info("No voice filename detected — using default file if available.")
            use_default = True
        else:
            st.sidebar.info(f"Recognized filename phrase: '{file_choice}'")
            if "use default" in file_choice or "default" in file_choice:
                use_default = True
            else:
                # naive normalization: turn spoken spaces/underscore into filename
                normalized = file_choice.replace(" underscore ", "_").replace(" space ", " ").replace(" ", "_")
                candidate = os.path.join("data", normalized.strip() + ".pdf")
                if os.path.exists(candidate):
                    filepath = candidate
                    st.sidebar.success(f"Found file: {candidate}")
                else:
                    st.sidebar.warning(f"Could not find file {candidate}. Falling back to default.")
                    use_default = True

    if use_default:
        if os.path.exists(DEFAULT_LOCAL_PDF):
            filepath = DEFAULT_LOCAL_PDF
            st.sidebar.success(f"Using default file: {os.path.basename(DEFAULT_LOCAL_PDF)}")
        else:
            st.error("No PDF available (no upload, no default). Please upload a PDF via the sidebar.")
            return

    # show PDF in UI
    display_pdf(filepath)

    # prepare pdf reader and metadata
    pdf_reader = PyPDF2.PdfReader(open(filepath, "rb"))
    total_pages = len(pdf_reader.pages)
    current_page_idx = 0

    # State variables
    querying = False

    st.sidebar.info("You can say: 'search', 'summarize', 'read aloud', 'start navigation', 'query', 'help', or 'exit'.")

    # Main listening loop (keeps listening until 'exit' command)
    while True:
        speak("Listening for a command...", speaker)
        command = listen_once(recognizer, timeout=8, phrase_time_limit=8)
        if not command:
            # no input this iteration; show hint and continue
            st.sidebar.info("No command detected. Say 'help' for commands or 'exit' to quit.")
            continue

        st.sidebar.info(f"Command: {command}")

        # Exit conditions
        if any(k in command for k in ["exit", "quit", "speaklink exit", "speak link exit", "stop assistant", "stop program"]):
            speak("Stopping assistant. Goodbye.", speaker)
            break

        # Help
        if "help" in command:
            help_text = (
                "Available voice commands: "
                "1) 'search' — search a word in the PDF. "
                "2) 'summarize' — summarize document. "
                "3) 'read aloud' — read the whole document aloud (you can say 'stop' to interrupt). "
                "4) 'start navigation' — control pages: 'next page', 'previous page', or 'page N'. "
                "5) 'query' — ask a question about the document contents. "
                "6) 'exit' — stop the assistant."
            )
            speak(help_text, speaker)
            continue

        # Search
        if "search" in command:
            speak("Please say the word or phrase you want to search for.", speaker)
            search_term = listen_once(recognizer, timeout=6, phrase_time_limit=6)
            if not search_term:
                speak("No search term detected.", speaker)
                continue
            pages = search_word_in_pdf(filepath, search_term)
            if pages:
                msg = f"Found '{search_term}' on page(s): {', '.join(map(str, pages))}"
                speak(msg, speaker)
            else:
                speak(f"'{search_term}' not found in document.", speaker)
            continue

        # Summarize
        if "summarize" in command or "summarise" in command:
            speak("Generating a summary. This may take a few seconds.", speaker)
            try:
                summary = llm_summarize(filepath)
                st.success("Summary:")
                st.write(summary)
                speak("Reading summarized content now.", speaker)
                speak(summary, speaker)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                speak("Sorry, I could not summarize the document.", speaker)
            continue

        # Read aloud (full document)
        if "read aloud" in command or "read the document" in command:
            speak("Reading document aloud. Say 'stop' to interrupt.", speaker)
            stop_requested = False
            for p_idx in range(total_pages):
                page_text = pdf_reader.pages[p_idx].extract_text() or ""
                sentences = sentence_split(page_text)
                for s in sentences:
                    speaker.say(s)
                    speaker.runAndWait()
                    # quick non-blocking listen to check for 'stop'
                    maybe = listen_once(recognizer, timeout=0.8, phrase_time_limit=1)
                    if maybe and "stop" in maybe:
                        speak("Stopping read aloud.", speaker)
                        stop_requested = True
                        break
                if stop_requested:
                    break
            continue

        # Navigation
        if "start navigation" in command or "navigation" in command:
            speak("Navigation mode: say 'next page', 'previous page', 'page N', or 'stop navigation' to exit navigation.", speaker)
            while True:
                nav_cmd = listen_once(recognizer, timeout=8, phrase_time_limit=6)
                if not nav_cmd:
                    speak("No navigation command detected. Say 'stop navigation' to exit.", speaker)
                    continue
                st.sidebar.info(f"Nav command: {nav_cmd}")
                if "next page" in nav_cmd:
                    current_page_idx = min(current_page_idx + 1, total_pages - 1)
                elif "previous page" in nav_cmd:
                    current_page_idx = max(current_page_idx - 1, 0)
                elif "page" in nav_cmd:
                    pn = extract_page_number(nav_cmd)
                    if pn and 1 <= pn <= total_pages:
                        current_page_idx = pn - 1
                    else:
                        speak(f"Invalid page number. Please say a page between 1 and {total_pages}.", speaker)
                        continue
                elif "stop navigation" in nav_cmd or "exit navigation" in nav_cmd:
                    speak("Exiting navigation mode.", speaker)
                    break
                else:
                    speak("I didn't understand that navigation command.", speaker)
                    continue

                # read the selected page aloud (short excerpt)
                page_text = pdf_reader.pages[current_page_idx].extract_text() or ""
                st.info(f"Page {current_page_idx + 1}/{total_pages}")
                st.write(page_text[:1000])  # show first part in UI
                speak(f"Reading page {current_page_idx + 1}", speaker)
                # read page sentences but stop if 'stop' detected
                for sent in sentence_split(page_text):
                    speaker.say(sent)
                    speaker.runAndWait()
                    # check for stop
                    maybe = listen_once(recognizer, timeout=0.6, phrase_time_limit=0.6)
                    if maybe and "stop" in maybe:
                        speak("Stopped reading page.", speaker)
                        break
            continue

        # Query (Q&A)
        if "query" in command or "ask" in command:
            speak("Entering query mode. Ask your question, or say 'stop' to exit query mode.", speaker)
            while True:
                question = listen_once(recognizer, timeout=10, phrase_time_limit=8)
                if not question:
                    speak("No question detected. Say 'stop' to exit query mode.", speaker)
                    continue
                if "stop" in question or "exit" in question:
                    speak("Exiting query mode.", speaker)
                    break
                # get document text and answer using T5
                doc_text = extract_text_from_pdf(filepath)
                try:
                    ans = answer_query_t5(question, doc_text)
                    st.success(f"Q: {question}")
                    st.write(f"A: {ans}")
                    speak(ans, speaker)
                except Exception as e:
                    st.error(f"Q&A failed: {e}")
                    speak("Sorry, I couldn't answer that question.", speaker)
            continue

        # Fallback
        speak("Command not recognized. Say 'help' for options.", speaker)

if __name__ == "__main__":
    main()
