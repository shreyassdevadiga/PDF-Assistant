import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import base64
import PyPDF2
import pyttsx3
import speech_recognition as sr
import os
import torch
import re
from speech_recognition import WaitTimeoutError, UnknownValueError

# ----------------- MODEL LOADING -----------------
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.float32
)

# ----------------- HELPER FUNCTIONS -----------------
def greet_user(recognizer):
    speaker = pyttsx3.init()
    greeting_message = (
        "Hello, I am SpeakLink, a virtual assistant at your service. "
        "You can give me commands as soon as I say 'listening for a command.' "
        "My capabilities include reading, summarizing, navigating, querying and searching. "
        "At any point, if you want help, you can say 'help'. "
        "To exit from the assistant, you can say 'speaklink exit'. "
        "I hope you have a pleasant experience with me!"
    )
    st.sidebar.info(greeting_message)
    speaker.say(greeting_message)
    speaker.runAndWait()

def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]["summary_text"]
    return result

def summarize_text_with_t5(text, max_len=150, min_len=30):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(
        input_text, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = base_model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def answer_query_t5(query, text):
    prompt = f"question: {query} context: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    answer_ids = base_model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        num_beams=3,
        early_stopping=True,
    )
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def answer_query(query, text):
    sentences = re.split(r"(?<=[^A-Z].[.?]) +", text)
    relevant_sentences = []
    for sentence in sentences:
        if query.lower() in sentence.lower():
            relevant_sentences.append(sentence.strip())
    return relevant_sentences

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def extract_page_number(command):
    try:
        words = command.split()
        page_index = words.index("page")
        page_number = int(words[page_index + 1])
        return page_number
    except (ValueError, IndexError):
        st.sidebar.warning("Invalid page command.")
        return None

def listen_for_command(recognizer, timeout=5, retries=3):
    # Do not listen while speaking summary or other long content
    if st.session_state.get("is_speaking", False):
        return None

    for attempt in range(retries):
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                if attempt == 0:
                    pyttsx3.speak("Listening for a command...")
                st.sidebar.info("Listening for a command...")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                if command:
                    st.sidebar.info(f"Command: {command}")
                    return command
            except sr.WaitTimeoutError:
                if attempt < retries - 1:
                    st.sidebar.info("No command detected. Please speak now...")
                    continue
                st.sidebar.warning("No command received. Please try again.")
                return None
            except sr.UnknownValueError:
                if attempt < retries - 1:
                    continue
                st.sidebar.warning("Could not understand audio. Please try again.")
                pyttsx3.speak("I couldn't understand that. Please try again.")
            except sr.RequestError as e:
                st.sidebar.error(f"Speech recognition error: {e}")
                pyttsx3.speak("Sorry, I'm having trouble with the speech service.")
                return None
            except Exception as e:
                st.sidebar.error(f"Unexpected error: {str(e)}")
                return None
    return None

def listen_for_command1(recognizer):
    # Do not listen while speaking summary or other long content
    if st.session_state.get("is_speaking", False):
        return "none"

    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=1)
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Command: {command}")
                return command
            except WaitTimeoutError:
                return "none"
            except UnknownValueError:
                return "none"

def listen_for_question(recognizer):
    while True:
        with sr.Microphone() as source:
            pyttsx3.speak("Listening for your question...")
            st.sidebar.info("Listening for your question...")
            audio = recognizer.listen(source, timeout=10)
            try:
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Question: {command}")
                return command
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand audio. Please try again.")
                pyttsx3.speak("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                st.sidebar.warning(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
                pyttsx3.speak(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
            continue

def read_page(pdf_reader, page_number, speaker, recognizer):
    page = pdf_reader.pages[page_number]
    text = page.extract_text() or ""
    sentences = text.split(".")
    for sentence in sentences:
        if sentence.strip():
            speaker.say(sentence)
            speaker.runAndWait()
        stop_command = listen_for_command1(recognizer)
        if stop_command and "stop" in stop_command:
            st.sidebar.info("Stopping reading...")
            pyttsx3.speak("Stopping reading...")
            return

def read_summarized_content(summary, speaker):
    # Block listening while speaking summary
    st.session_state.is_speaking = True
    try:
        speaker.say("Reading the summarised content.")
        speaker.say(summary)
        speaker.runAndWait()
    finally:
        st.session_state.is_speaking = False

def listen_for_search_word(recognizer):
    while True:
        with sr.Microphone() as source:
            pyttsx3.speak("Please speak the word you want to search for.")
            st.sidebar.info("Please speak the word you want to search for.")
            audio = recognizer.listen(source, timeout=10)
            try:
                search_word = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Search word: {search_word}")
                return search_word
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand the word. Please try again.")
                pyttsx3.speak("Could not understand the word. Please try again.")
            except sr.RequestError as e:
                st.sidebar.warning(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
                pyttsx3.speak(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
            continue

def search_word_in_pdf(filepath, search_word):
    pdf_reader = PyPDF2.PdfReader(filepath)
    total_pages = len(pdf_reader.pages)
    occurrences = []
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        text = (page.extract_text() or "").lower()
        if search_word in text:
            occurrences.append(page_num + 1)
    return occurrences

def listen_for_note(recognizer):
    with sr.Microphone() as source:
        pyttsx3.speak("Please speak your note.")
        st.sidebar.info("Please speak your note.")
        audio = recognizer.listen(source, timeout=10)
        try:
            note = recognizer.recognize_google(audio).lower()
            st.sidebar.info(f"Note: {note}")
            return note
        except sr.UnknownValueError:
            st.sidebar.warning("Could not understand the note. Please try again.")
            pyttsx3.speak("Could not understand the note. Please try again.")
            return None
        except sr.RequestError as e:
            st.sidebar.warning(f"Error from speech service: {e}")
            pyttsx3.speak("Sorry, there was an error with the speech service.")
            return None

def add_highlight(page_index, text_snippet):
    page_index = int(page_index)
    if page_index not in st.session_state.highlights:
        st.session_state.highlights[page_index] = []
    st.session_state.highlights[page_index].append(text_snippet)
    st.sidebar.success(f"Highlighted on page {page_index + 1}.")
    pyttsx3.speak(f"Highlighted on page {page_index + 1}.")

def add_annotation(page_index, note_text):
    page_index = int(page_index)
    if page_index not in st.session_state.annotations:
        st.session_state.annotations[page_index] = []
    st.session_state.annotations[page_index].append(note_text)
    st.sidebar.success(f"Note added on page {page_index + 1}.")
    pyttsx3.speak(f"Note added on page {page_index + 1}.")

# ----------------- MAIN APP -----------------
st.set_page_config(layout="wide")

def main():
    st.title("SPEAKLINK : Voice Automated PDF Assistant for the Visually Impaired")

    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()

    if "highlights" not in st.session_state:
        st.session_state.highlights = {}
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "is_speaking" not in st.session_state:
        st.session_state.is_speaking = False

    greet_user(recognizer)

    querying = False

    # -------- Get file name by voice --------
    while True:
        with sr.Microphone() as source:
            st.sidebar.info("Listening for the file name...")
            pyttsx3.speak("Listening for the file name...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                st.sidebar.warning("No speech detected. Please try again.")
                pyttsx3.speak("I didn't hear anything. Please try again.")
                continue

        try:
            file_name = recognizer.recognize_google(audio)
            file_name = file_name.replace("underscore", "_").replace(" ", "").strip()
            file_name = file_name.replace("space", " ")
            st.sidebar.info(f"File name recognized: {file_name}")

            if file_name.lower() == "quit":
                st.sidebar.info("Exiting the program...")
                pyttsx3.speak("Exiting the program...")
                return

            filepath = os.path.join("data", file_name.strip() + ".pdf")

            if os.path.exists(filepath):
                pyttsx3.speak("File uploaded successfully")
                break
            else:
                st.sidebar.error(
                    f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' to exit."
                )
                pyttsx3.speak(
                    f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' to exit."
                )

        except sr.UnknownValueError:
            st.sidebar.error("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            st.sidebar.error(
                f"Could not request results from Google Speech Recognition service; {e}"
            )

    pdf_reader = PyPDF2.PdfReader(filepath)
    total_pages = len(pdf_reader.pages)
    current_page = 0
    full_text = extract_text_from_pdf(filepath)

    col1, col2 = st.columns(2)

    with col1:
        st.sidebar.info("Uploaded File")
        displayPDF(filepath)

        st.markdown("### Highlights and Notes")
        for p, items in st.session_state.highlights.items():
            st.markdown(f"**Page {p + 1} highlights:**")
            for i, snip in enumerate(items, 1):
                st.write(f"{i}. {snip[:150]}...")

        for p, notes in st.session_state.annotations.items():
            st.markdown(f"**Page {p + 1} notes:**")
            for i, note in enumerate(notes, 1):
                st.write(f"{i}. {note}")

    with col2:
        while True:
            command = listen_for_command(recognizer, timeout=5, retries=3)
            if not command:
                continue

            # -------- SEARCH WORD --------
            if "search" in command:
                search_word = listen_for_search_word(recognizer)
                st.sidebar.info(f"Searching for: {search_word}")
                page_numbers = search_word_in_pdf(filepath, search_word)
                if page_numbers:
                    pages_str = ", ".join(map(str, page_numbers))
                    st.sidebar.success(
                        f"Word '{search_word}' found on page(s): {pages_str}"
                    )
                    pyttsx3.speak(
                        f"Word '{search_word}' found on page(s): {pages_str}"
                    )
                else:
                    st.sidebar.warning(
                        f"Word '{search_word}' not found in the document."
                    )
                    pyttsx3.speak(
                        f"Word '{search_word}' not found in the document."
                    )

            # -------- QUERY / QA --------
            elif "query" in command:
                if not querying:
                    st.sidebar.info("Starting querying feature. Please ask your question.")
                    pyttsx3.speak("Starting querying feature. Please ask your question.")
                    querying = True

                while querying:
                    question = listen_for_question(recognizer)
                    if "stop" in question:
                        st.sidebar.info("Stopping querying feature.")
                        pyttsx3.speak("Stopping querying feature.")
                        querying = False
                        break
                    else:
                        answer = answer_query_t5(question, full_text)
                        st.success(f"Answer: {answer}")
                        pyttsx3.speak(answer)

            # -------- PAGE SUMMARY --------
            elif "summarize page" in command or "summarise page" in command:
                page_text = pdf_reader.pages[current_page].extract_text() or ""
                if not page_text.strip():
                    st.sidebar.warning("No text found on this page to summarize.")
                    pyttsx3.speak("No text found on this page to summarize.")
                else:
                    st.sidebar.info(f"Summarizing page {current_page + 1}...")
                    pyttsx3.speak(
                        f"Summarizing page {current_page + 1}. Please wait."
                    )
                    page_summary = summarize_text_with_t5(page_text)
                    st.success(page_summary)
                    read_summarized_content(page_summary, speaker)

            # -------- DOCUMENT SUMMARY --------
            elif "summarize" in command or "summarise" in command:
                st.sidebar.info("Summarizing document...")
                pyttsx3.speak("Summarizing the document. Please wait.")
                try:
                    summary = summarize_text_with_t5(full_text)
                    st.success(summary)
                    read_summarized_content(summary, speaker)
                except Exception as e:
                    st.sidebar.error(f"Error while summarizing: {e}")
                    pyttsx3.speak(
                        "Sorry, there was an error while summarizing the document."
                    )

            # -------- READ ALOUD ALL PAGES --------
            elif "read aloud" in command:
                st.sidebar.info("Reading aloud...")
                page_num = 0
                while page_num < len(pdf_reader.pages):
                    from_page = pdf_reader.pages[page_num]
                    text = from_page.extract_text() or ""
                    sentences = text.split(".")
                    stop_flag = False
                    for sentence in sentences:
                        if sentence.strip():
                            speaker.say(sentence)
                            speaker.runAndWait()
                        cmd = listen_for_command1(recognizer)
                        if cmd and "stop" in cmd:
                            st.sidebar.info("Stopping reading...")
                            pyttsx3.speak("Stopping reading...")
                            stop_flag = True
                            break
                    if stop_flag:
                        break
                    else:
                        page_num += 1

            # -------- NAVIGATION MODE --------
            elif "navigation" in command or "start navigation" in command:
                st.sidebar.info(
                    "Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'."
                )
                pyttsx3.speak(
                    "Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'."
                )

                while True:
                    nav_command = listen_for_command(recognizer, timeout=5, retries=2)
                    if nav_command is None:
                        st.sidebar.warning(
                            "No valid command received. Please try again or say 'stop navigation' to exit."
                        )
                        pyttsx3.speak(
                            "No valid command received. Please try again or say 'stop navigation' to exit."
                        )
                        continue

                    try:
                        nav_command = nav_command.lower()

                        if "next page" in nav_command:
                            current_page = min(current_page + 1, total_pages - 1)
                            st.sidebar.info(f"Moving to page {current_page + 1}")
                            pyttsx3.speak(f"Moving to page {current_page + 1}")

                        elif "previous page" in nav_command:
                            current_page = max(current_page - 1, 0)
                            st.sidebar.info(f"Moving to page {current_page + 1}")
                            pyttsx3.speak(f"Moving to page {current_page + 1}")

                        elif "page" in nav_command:
                            page_number = extract_page_number(nav_command)
                            if page_number is not None and 1 <= page_number <= total_pages:
                                current_page = page_number - 1
                                st.sidebar.info(f"Moving to page {page_number}")
                                pyttsx3.speak(f"Moving to page {page_number}")
                            else:
                                st.sidebar.warning(
                                    f"Invalid page number. Please enter a number between 1 and {total_pages}."
                                )
                                pyttsx3.speak(
                                    f"Invalid page number. Please enter a number between 1 and {total_pages}."
                                )
                                continue

                        elif "stop navigation" in nav_command:
                            st.sidebar.info("Exiting navigation mode.")
                            pyttsx3.speak("Exiting navigation mode.")
                            break

                        else:
                            st.sidebar.warning(
                                "I didn't understand that command. Please say 'Next Page', 'Previous Page', 'Page X', or 'Stop Navigation'."
                            )
                            pyttsx3.speak(
                                "I didn't understand that command. Please say 'Next Page', 'Previous Page', 'Page X', or 'Stop Navigation'."
                            )
                            continue

                        read_page(pdf_reader, current_page, speaker, recognizer)

                    except Exception as e:
                        st.sidebar.error(f"Error processing navigation: {str(e)}")
                        pyttsx3.speak(
                            "Sorry, there was an error processing your command."
                        )
                        continue

            # -------- HIGHLIGHT CURRENT PAGE --------
            elif "highlight" in command:
                page_text = pdf_reader.pages[current_page].extract_text() or ""
                snippet = page_text[:200]
                add_highlight(current_page, snippet)

            # -------- ADD NOTE / ANNOTATION --------
            elif "note" in command or "annotation" in command:
                note_text = listen_for_note(recognizer)
                if note_text:
                    add_annotation(current_page, note_text)

            # -------- HELP --------
            elif "help" in command:
                st.sidebar.info("Please let me know with which command I can help you!")
                pyttsx3.speak("Please let me know with which command I can help you!")

                while True:
                    hcommand = listen_for_command(recognizer)
                    if not hcommand:
                        continue

                    if "read page" in hcommand:
                        st.sidebar.info(
                            "To read a specific page, say 'start navigation' followed by the page number."
                        )
                        pyttsx3.speak(
                            "To read a specific page, say 'start navigation' followed by the page number."
                        )
                        continue

                    if "read" in hcommand:
                        st.sidebar.info("To read aloud the entire file, say 'read aloud'.")
                        pyttsx3.speak(
                            "To read aloud the entire file, say 'read aloud'."
                        )
                        continue

                    if "summarize page" in hcommand:
                        st.sidebar.info(
                            "To summarize only the current page, say 'summarize page'."
                        )
                        pyttsx3.speak(
                            "To summarize only the current page, say 'summarize page'."
                        )
                        continue

                    if "summarize" in hcommand:
                        st.sidebar.info(
                            "To summarize the entire file, say 'summarize'."
                        )
                        pyttsx3.speak(
                            "To summarize the entire file, say 'summarize'."
                        )
                        continue

                    if "search word" in hcommand:
                        st.sidebar.info(
                            "To search for a specific word in the file, say 'search' and then speak the word."
                        )
                        pyttsx3.speak(
                            "To search for a specific word in the file, say search and then speak the word."
                        )
                        continue

                    if "query" in hcommand:
                        st.sidebar.info(
                            "To query the file, say 'query' and then ask your question."
                        )
                        pyttsx3.speak(
                            "To query the file, say query and then ask your question."
                        )
                        continue

                    if "navigation" in hcommand:
                        st.sidebar.info(
                            "Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'."
                        )
                        pyttsx3.speak(
                            "Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'."
                        )
                        continue

                    if "highlight" in hcommand:
                        st.sidebar.info(
                            "To highlight the current page, say 'highlight'."
                        )
                        pyttsx3.speak(
                            "To highlight the current page, say highlight."
                        )
                        continue

                    if "note" in hcommand or "annotation" in hcommand:
                        st.sidebar.info(
                            "To add a note on the current page, say 'note' and then speak your note."
                        )
                        pyttsx3.speak(
                            "To add a note on the current page, say note and then speak your note."
                        )
                        continue

                    if any(
                        phrase in hcommand for phrase in ["exit help", "stop help"]
                    ):
                        st.sidebar.info("Stopping help feature.")
                        pyttsx3.speak("Stopping help feature.")
                        break

                    else:
                        st.sidebar.info("Invalid command, please try again")
                        pyttsx3.speak("Invalid command, please try again")
                        continue

            # -------- STOP / EXIT ASSISTANT --------
            elif any(
                phrase in command
                for phrase in [
                    "stop",
                    "speak link stop",
                    "speaklink stop",
                    "speaking stop",
                    "speaklink exit",
                    "speak link exit",
                    "speaking exit",
                ]
            ):
                st.sidebar.info("Stopping assistant.")
                pyttsx3.speak("Stopping assistant.")
                break

            # -------- INVALID COMMAND --------
            else:
                st.sidebar.warning("Invalid command.")
                pyttsx3.speak("Invalid command.")
                continue

if __name__ == "__main__":
    main()
