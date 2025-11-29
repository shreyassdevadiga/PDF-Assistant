
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import base64
import PyPDF2
import pyttsx3
import speech_recognition as sr
import os
import torch
import re
import time
from speech_recognition import WaitTimeoutError , UnknownValueError

# Model and tokenizer loading
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

def init_speaker():
    """Initialize TTS engine - returns a fresh engine"""
    try:
        engine = pyttsx3.init()
        reading_speed = st.session_state.get("reading_speed", 150)
        engine.setProperty("rate", reading_speed)
        engine.setProperty("volume", 1.0)
        return engine
    except Exception as e:
        print(f"[TTS INIT ERROR]: {e}")
        return None

def speak_text(text, speaker=None, interruptible=False):
    """Speak text using Windows SAPI or pyttsx3"""
    try:
        print(f"[SPEAKING]: {text[:100]}")  # Debug
        
        # Try using Windows SAPI directly (more reliable in Streamlit)
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            # Use PowerShell to speak via Windows SAPI
            reading_speed = st.session_state.get("reading_speed", 150)
            # Convert speed (100-250) to rate (-10 to 10)
            rate = int((reading_speed - 150) / 15)
            
            # Escape single quotes in text
            text_escaped = text.replace("'", "''")
            
            ps_command = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.Rate = {rate}
            $synth.Volume = 100
            $synth.Speak("{text_escaped}")
            '''
            
            if interruptible:
                # Store process so it can be killed
                process = subprocess.Popen(["powershell", "-Command", ps_command], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
                st.session_state["tts_process"] = process
                process.wait(timeout=30)
            else:
                subprocess.run(["powershell", "-Command", ps_command], 
                              capture_output=True, timeout=30)
        else:
            # Fallback to pyttsx3 for non-Windows
            engine = pyttsx3.init()
            reading_speed = st.session_state.get("reading_speed", 150)
            engine.setProperty("rate", reading_speed)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
        
        print("[SPEAKING]: Done")  # Debug
        time.sleep(0.1)
    except Exception as e:
        print(f"[TTS ERROR]: {e}")
        import traceback
        traceback.print_exc()

def stop_speaking():
    """Stop any ongoing TTS"""
    try:
        if "tts_process" in st.session_state:
            process = st.session_state["tts_process"]
            if process and process.poll() is None:  # Process is still running
                process.terminate()
                process.wait(timeout=2)
                print("[TTS]: Stopped")
    except Exception as e:
        print(f"[TTS STOP ERROR]: {e}")

# Function to greet the user
def greet_user(speaker):
    greeting_message = (
            "Hello, I am SpeakLink, a virtual assistant at your service. "
            "You can give me commands as soon as I say listening for a command. "
            "My capabilities include reading, summarizing, navigating, querying and searching. "
            "At any point, if you want help, you can say help. "
            "To exit from the assistant, you can say speaklink exit. "
            "I hope you have a pleasant experience with me!"
        )

    st.sidebar.info(greeting_message)
    speak_text(greeting_message, speaker)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

def answer_query_t5(query, text):
    prompt = f"question: {query} context: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    answer_ids = base_model.generate(inputs, max_length=200, num_return_sequences=1, num_beams=3, early_stopping=True)
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer

# Function to summarize text using T5 model
def summarize_text_with_t5(text, max_len=150, min_len=30):
    """Summarize text using T5 model"""
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = base_model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to extract text from the PDF file
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to answer user's query based on extracted text
def answer_query(query, text):
    # Split text into sentences
    sentences = re.split(r'(?<=[^A-Z].[.?]) +', text)

    # Find the most relevant sentence containing the query
    relevant_sentences = []
    for sentence in sentences:
        if query.lower() in sentence.lower():
            relevant_sentences.append(sentence.strip())

    return relevant_sentences

@st.cache_data
# Function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Shreya code
def extract_page_number(command):
    try:
        words = command.split()
        page_index = words.index("page")
        page_number = int(words[page_index + 1])
        return page_number
    except (ValueError, IndexError):
        st.sidebar.warning("Invalid page command.")
        speak_text("Invalid page command.", init_speaker())
        return None

def listen_for_command(recognizer, speaker, timeout=5, retries=3):
    for attempt in range(retries):
        # Speak BEFORE opening microphone
        if attempt == 0:  # Only speak on first attempt
            st.sidebar.info("Listening for a command...")
            speak_text("Listening for a command...", speaker)
            time.sleep(0.2)
        
        with sr.Microphone() as source:
            try:
                # Adjust for ambient noise for better recognition
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with shorter timeouts for more responsive behavior
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                
            except sr.WaitTimeoutError:
                audio = None
            except Exception as e:
                st.sidebar.error(f"Unexpected error: {str(e)}")
                return None
        
        # Microphone is now closed, safe to speak and process
        if audio is None:
            if attempt < retries - 1:
                st.sidebar.info("No command detected. Please speak now...")
                speak_text("No command detected. Please speak now.", speaker)
                continue
            else:
                st.sidebar.warning("No command received. Please try again.")
                speak_text("No command received. Please try again.", speaker)
                return None
        
        try:
            command = recognizer.recognize_google(audio).lower()
            if command:
                st.sidebar.info(f"Command: {command}")
                return command
                
        except sr.UnknownValueError:
            if attempt < retries - 1:
                continue
            st.sidebar.warning("Could not understand audio. Please try again.")
            speak_text("Could not understand audio. Please try again.", speaker)
            return None
            
        except sr.RequestError as e:
            st.sidebar.error(f"Speech recognition error: {e}")
            speak_text("Sorry, I'm having trouble with the speech service.", speaker)
            return None
    
    return None  # If all retries fail

def listen_for_command1(recognizer):
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=1)
            command = recognizer.recognize_google(audio).lower()
            st.sidebar.info(f"Command: {command}")
            return command
        except WaitTimeoutError:
            return None
        except UnknownValueError:
            return None
        except Exception:
            return None
            
def listen_for_question(recognizer, speaker):
    while True:
        with sr.Microphone() as source:
            speak_text("Listening for your question...", speaker)
            st.sidebar.info("Listening for your question...")
            try:
                audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Question: {command}")
                speak_text(f"You asked: {command}", speaker)
                return command
            except sr.WaitTimeoutError:
                st.sidebar.warning("No speech detected. Please try again.")
                speak_text("No speech detected. Please try again.", speaker)
                return None
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand audio. Please try again.")
                speak_text("Could not understand audio. Please try again.", speaker)
            except sr.RequestError as e:
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                speak_text("Sorry, there was an error with the speech service.", speaker)
                return None
            continue
            
def read_page(pdf_reader, page_number, speaker, recognizer):
    page = pdf_reader.pages[page_number]
    text = page.extract_text()
    
    if not text or not text.strip():
        speak_text("This page appears to be empty.", speaker)
        return False  # Return False to indicate not stopped by user

    sentences = text.split('.')  # Split text into sentences
    for sentence in sentences:
        if sentence.strip():
            # Split long sentences into smaller chunks (every 10 words)
            words = sentence.strip().split()
            chunk_size = 10
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                speak_text(chunk, speaker, interruptible=True)
                
                # Check if stop command received during reading
                stop_command = listen_for_command1(recognizer)
                if stop_command and "stop" in stop_command:
                    stop_speaking()  # Stop any ongoing speech
                    st.sidebar.info("Stopping reading...")
                    speak_text("Stopping reading...", speaker)
                    return True  # Return True to indicate stopped by user
    
    return False  # Completed without interruption



def read_summarized_content(summary, speaker):
    # Split summary into chunks for better pacing
    chunks = re.split(r'(?<=[.!?]) +', summary)
    for chunk in chunks:
        if chunk.strip():
            speak_text(chunk.strip(), speaker)

# Function to listen for the word to be searched
def listen_for_search_word(recognizer, speaker):
    while True:
        with sr.Microphone() as source:
            speak_text("Please speak the word you want to search for.", speaker)
            st.sidebar.info("Please speak the word you want to search for.")
            try:
                audio = recognizer.listen(source, timeout=10)
                search_word = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Search word: {search_word}")
                speak_text(f"Searching for: {search_word}", speaker)
                return search_word
            except sr.WaitTimeoutError:
                st.sidebar.warning("No speech detected. Please try again.")
                speak_text("No speech detected. Please try again.", speaker)
                return None
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand the word. Please try again.")
                speak_text("Could not understand the word. Please try again.", speaker)
            except sr.RequestError as e:
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                speak_text("Sorry, there was an error with the speech service.", speaker)
                return None
            continue

# Function to search for the word in the PDF and return page numbers
def search_word_in_pdf(filepath, search_word):
    pdf_reader = PyPDF2.PdfReader(filepath)
    total_pages = len(pdf_reader.pages)
    occurrences = []
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text and search_word in text.lower():
            occurrences.append(page_num + 1)  # Adding 1 to convert to 1-based indexing
    return occurrences

# Function to change reading speed
def change_reading_speed(command, speaker):
    """Adjust the reading speed based on voice command"""
    current_speed = st.session_state.get("reading_speed", 150)
    
    if "faster" in command or "fast" in command:
        new_speed = min(current_speed + 30, 250)  # Max speed 250
        st.session_state["reading_speed"] = new_speed
        st.sidebar.success(f"Reading speed increased to {new_speed}")
        speak_text(f"Reading speed increased", speaker)
        return True
    elif "slower" in command or "slow" in command:
        new_speed = max(current_speed - 30, 100)  # Min speed 100
        st.session_state["reading_speed"] = new_speed
        st.sidebar.success(f"Reading speed decreased to {new_speed}")
        speak_text(f"Reading speed decreased", speaker)
        return True
    elif "normal" in command or "default" in command:
        new_speed = 150
        st.session_state["reading_speed"] = new_speed
        st.sidebar.success(f"Reading speed set to normal")
        speak_text(f"Reading speed set to normal", speaker)
        return True
    return False


# Streamlit code    
st.set_page_config(layout="wide")

def main():
    st.title("SPEAKLINK : Voice Automated PDF  Assistant for the Visually Impaired")

    recognizer = sr.Recognizer()
    speaker = init_speaker()
    
    greet_user(speaker)

    querying = False

    # Speech recognition to capture the file name from user's voice
    while True:
        # Speak BEFORE opening microphone
        st.sidebar.info("Listening for the file name...")
        speak_text("Listening for the file name...", speaker)
        time.sleep(0.2)  # Small delay before opening mic
        
        with sr.Microphone() as source:
            try:
                # Adjust for ambient noise and set a timeout
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                st.sidebar.warning("No speech detected. Please try again.")
                continue  # Skip to next iteration
        
        # Microphone is now closed, safe to speak
        try:
            file_name = recognizer.recognize_google(audio)
            file_name = file_name.replace('underscore', '_').replace(' ', '').strip()
            file_name = file_name.replace('space', ' ')
            st.sidebar.info(f"File name recognized: {file_name}")
            speak_text(f"File name recognized: {file_name}", speaker)

            if file_name.lower() == "quit":
                st.sidebar.info("Exiting the program...")
                speak_text("Exiting the program...", speaker)
                return

            # Appending ".pdf" to the file name
            filepath = os.path.join("data", file_name.strip() + ".pdf")

            # Checking if the file exists
            if os.path.exists(filepath):
                speak_text("File uploaded successfully", speaker)
                st.sidebar.success("Uploaded File")
                break
            else:
                st.sidebar.error(f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' to exit.")
                speak_text(f"File {file_name} dot pdf does not exist. Please try again or say quit to exit.", speaker)

        except sr.UnknownValueError:
            st.sidebar.error("Sorry, could not understand the audio.")
            speak_text("Sorry, could not understand the audio.", speaker)
            continue
        except sr.RequestError as e:
            st.sidebar.error(f"Could not request results from Google Speech Recognition service; {e}")
            speak_text("Sorry, there was an error with the speech service.", speaker)
            continue

    # Reading PDF file
    pdf_reader = PyPDF2.PdfReader(filepath)
    total_pages = len(pdf_reader.pages)
    current_page = 0

    col1, col2 = st.columns(2)

    with col1:
            st.sidebar.success("File uploaded successfully. Ready for commands.")
            speak_text("File uploaded successfully. Ready for commands.", speaker)
            displayPDF(filepath)

    with col2:
            while True:
                command = listen_for_command(recognizer, speaker, timeout=5, retries=3)
                
                if command is None:
                    continue

                # Check for reading speed control
                if "speed" in command or "faster" in command or "slower" in command:
                    change_reading_speed(command, speaker)
                    continue

                elif "search" in command:
                    search_word = listen_for_search_word(recognizer, speaker)
                    if search_word is None:
                        continue
                    page_numbers = search_word_in_pdf(filepath, search_word)
                    if page_numbers:
                        pages_str = ', '.join(map(str, page_numbers))
                        st.sidebar.success(f"Word '{search_word}' found on page(s): {pages_str}")
                        speak_text(f"Word {search_word} found on page {pages_str}", speaker)
                    else:
                        st.sidebar.warning(f"Word '{search_word}' not found in the document.")
                        speak_text(f"Word {search_word} not found in the document.", speaker)


                elif "query" in command:
                    if not querying:
                        st.sidebar.info("Starting querying feature. Please ask your question.")
                        speak_text("Starting querying feature. Please ask your question.", speaker)
                        querying = True

                    while querying:
                        question = listen_for_question(recognizer, speaker)
                        if question is None:
                            break
                        if "stop" in question:
                            st.sidebar.info("Stopping querying feature.")
                            speak_text("Stopping querying feature.", speaker)
                            querying = False
                            break
                        else:
                            st.sidebar.info("Processing your question...")
                            speak_text("Processing your question. Please wait.", speaker)
                            text = extract_text_from_pdf(filepath)
                            answer = answer_query_t5(question, text)
                            st.success(f"Answer: {answer}")
                            speak_text(f"The answer is: {answer}", speaker)


                elif "summarize page" in command or "summarise page" in command:
                    # Summarize current page only
                    page_text = pdf_reader.pages[current_page].extract_text()
                    if not page_text or not page_text.strip():
                        st.sidebar.warning("No text found on this page to summarize.")
                        speak_text("No text found on this page to summarize.", speaker)
                    else:
                        st.sidebar.info(f"Summarizing page {current_page + 1}...")
                        speak_text(f"Summarizing page {current_page + 1}. Please wait.", speaker)
                        page_summary = summarize_text_with_t5(page_text)
                        st.success(f"Summary: {page_summary}")
                        speak_text("Here is the summary:", speaker)
                        read_summarized_content(page_summary, speaker)

                elif "summarize" in command or "summarise" in command:
                    # Summarize entire document
                    st.sidebar.info("Summarizing entire document...")
                    speak_text("Summarizing the document. Please wait.", speaker)
                    try:
                        text = extract_text_from_pdf(filepath)
                        summary = summarize_text_with_t5(text, max_len=500, min_len=100)
                        st.success(f"Summary: {summary}")
                        speak_text("Here is the document summary:", speaker)
                        read_summarized_content(summary, speaker)
                    except Exception as e:
                        st.sidebar.error(f"Error while summarizing: {e}")
                        speak_text("Sorry, there was an error while summarizing the document.", speaker)

                elif "read aloud" in command:
                    st.sidebar.info("Reading aloud the entire document...")
                    speak_text("Reading aloud the entire document. Say stop to pause.", speaker)
                    page_num = 0
                    stop_flag = False
                    
                    while page_num < len(pdf_reader.pages) and not stop_flag:
                        from_page = pdf_reader.pages[page_num]
                        text = from_page.extract_text()
                        
                        if text and text.strip():
                            sentences = text.split('.')  # Split text into sentences
                            for sentence in sentences:
                                if sentence.strip():
                                    # Split into smaller chunks for better interruption
                                    words = sentence.strip().split()
                                    chunk_size = 10
                                    for i in range(0, len(words), chunk_size):
                                        chunk = ' '.join(words[i:i+chunk_size])
                                        speak_text(chunk, speaker, interruptible=True)
                                        
                                        # Check if stop command received during reading
                                        cmd = listen_for_command1(recognizer)
                                        if cmd and "stop" in cmd:
                                            stop_speaking()  # Stop any ongoing speech
                                            st.sidebar.info("Stopping reading...")
                                            speak_text("Stopping reading...", speaker)
                                            stop_flag = True
                                            break
                                    
                                    if stop_flag:
                                        break
                            
                            if not stop_flag:
                                page_num += 1
                        else:
                            page_num += 1



                elif "navigation" in command or "start navigation" in command:
                    st.sidebar.info("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                    speak_text("Starting navigation. You can say Next Page, Previous Page, or Page X.", speaker)

                    while True:
                        # Use the improved listen_for_command with timeout and retries
                        nav_command = listen_for_command(recognizer, speaker, timeout=5, retries=2)

                        if nav_command is None:
                            st.sidebar.warning("No valid command received. Please try again or say 'stop navigation' to exit.")
                            speak_text("No valid command received. Please try again or say stop navigation to exit.", speaker)
                            continue

                        try:
                            nav_command = nav_command.lower()
                            
                            if "next page" in nav_command:
                                current_page = min(current_page + 1, total_pages - 1)
                                st.sidebar.info(f"Moving to page {current_page + 1}")
                                speak_text(f"Moving to page {current_page + 1}", speaker)
                                
                            elif "previous page" in nav_command:
                                current_page = max(current_page - 1, 0)
                                st.sidebar.info(f"Moving to page {current_page + 1}")
                                speak_text(f"Moving to page {current_page + 1}", speaker)
                                
                            elif "page" in nav_command:
                                page_number = extract_page_number(nav_command)
                                if page_number is not None and 1 <= page_number <= total_pages:
                                    current_page = page_number - 1
                                    st.sidebar.info(f"Moving to page {page_number}")
                                    speak_text(f"Moving to page {page_number}", speaker)
                                else:
                                    st.sidebar.warning(f"Invalid page number. Please enter a number between 1 and {total_pages}.")
                                    speak_text(f"Invalid page number. Please enter a number between 1 and {total_pages}.", speaker)
                                    continue
                                    
                            elif "stop navigation" in nav_command:
                                st.sidebar.info("Exiting navigation mode.")
                                speak_text("Exiting navigation mode.", speaker)
                                break
                                
                            else:
                                st.sidebar.warning("I didn't understand that command. Please say 'Next Page', 'Previous Page', 'Page X', or 'Stop Navigation'.")
                                speak_text("I didn't understand that command.", speaker)
                                continue

                            # Read the current page
                            was_stopped = read_page(pdf_reader, current_page, speaker, recognizer)
                            if was_stopped:
                                # User said stop, exit navigation
                                break
                            
                        except Exception as e:
                            st.sidebar.error(f"Error processing navigation: {str(e)}")
                            speak_text("Sorry, there was an error processing your command.", speaker)
                            continue
                        
                elif "help" in command:
                    st.sidebar.info("Please let me know with which command I can help you!")
                    speak_text("Please let me know with which command I can help you!", speaker)
                    
                    while True:
                        hcommand = listen_for_command(recognizer, speaker)
                        
                        if hcommand is None:
                            continue

                        if "read page" in hcommand:
                            st.sidebar.info("To read a specific page, say 'start navigation' followed by the page number.")
                            speak_text("To read a specific page, say start navigation followed by the page number.", speaker)
                            continue

                        elif "read" in hcommand:
                            st.sidebar.info("To read aloud the entire file, say 'read aloud'.")
                            speak_text("To read aloud the entire file, say read aloud.", speaker)
                            continue

                        elif "summarize page" in hcommand or "summarise page" in hcommand:
                            st.sidebar.info("To summarize only the current page, say 'summarize page'.")
                            speak_text("To summarize only the current page, say summarize page.", speaker)
                            continue

                        elif "summarize" in hcommand or "summarise" in hcommand:
                            st.sidebar.info("To summarize the entire file, say 'summarize'.")
                            speak_text("To summarize the entire file, say summarize.", speaker)
                            continue

                        elif "search" in hcommand:
                            st.sidebar.info("To search for a specific word in the file, say 'search' followed by the word.")
                            speak_text("To search for a specific word in the file, say search followed by the word.", speaker)
                            continue

                        elif "query" in hcommand:
                            st.sidebar.info("To query the file, say 'query' followed by your question.")
                            speak_text("To query the file, say query followed by your question.", speaker)
                            continue

                        elif "navigation" in hcommand:
                            st.sidebar.info("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                            speak_text("Starting navigation. You can say Next Page, Previous Page, or Page X.", speaker)
                            continue

                        elif "speed" in hcommand or "faster" in hcommand or "slower" in hcommand:
                            st.sidebar.info("To change reading speed, say 'read faster', 'read slower', or 'normal speed'.")
                            speak_text("To change reading speed, say read faster, read slower, or normal speed.", speaker)
                            continue

                        elif any(phrase in hcommand for phrase in ["exit help", "stop help"]):
                            st.sidebar.info("Stopping help feature.")
                            speak_text("Stopping help feature.", speaker)
                            break
                        
                        else:
                            st.sidebar.info("Invalid command, please try again")
                            speak_text("Invalid command, please try again", speaker)
                            continue
                            
                       
                elif any(phrase in command for phrase in ["stop", "speak link stop", "speaklink stop", "speaking stop", "speaklink exit", "speak link exit", "speaking exit"]):
                    st.sidebar.info("Stopping assistant.")
                    speak_text("Stopping assistant.", speaker)
                    break
                
                else:
                    st.sidebar.warning("Invalid command.")
                    speak_text("Invalid command.", speaker)
                    continue
                
                
if __name__ == "__main__":
    main()
