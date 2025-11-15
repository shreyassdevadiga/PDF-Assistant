import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from speech_recognition import WaitTimeoutError , UnknownValueError

# Helper function for text-to-speech
def speak(text):
    """Convert text to speech using a fresh engine instance each time."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"TTS Error: {e}")

# Model and tokenizer loading
try:
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)
    st.success("Model loaded successfully!")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("The app will continue but summarization and query features may not work.")
    speak("Warning: AI model could not be loaded. Summarization and query features will not be available. Reading and navigation features will still work.")
    tokenizer = None
    base_model = None
    model_loaded = False

# Function to greet the user
def greet_user(recognizer):
    speaker = pyttsx3.init()

    greeting_message = (
            "Hello, I am SpeakLink, a virtual assistant at your service. "
            "You can give me commands as soon as I say 'listening for a command.'\n"
            "My capabilities include reading, summarizing, navigating, quering and searching. \n"
            "At any point, if you want help, you can say 'help'.\n"
            "To exit from the assistant, you can say 'speaklink exit'.\n"
            "I hope you have a pleasant experience with me!"
        )

    st.sidebar.info(greeting_message)
    speaker.say(greeting_message)
    speaker.runAndWait()

# File loader and preprocessing
def file_preprocessing(file, max_chars=3000):
    """Extract and preprocess text from PDF, limiting size to prevent memory issues."""
    try:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for text in texts:
            final_texts = final_texts + text.page_content
            # Limit text size to prevent memory issues
            if len(final_texts) >= max_chars:
                final_texts = final_texts[:max_chars]
                break
        return final_texts
    except Exception as e:
        st.error(f"Error preprocessing file: {str(e)}")
        speak("Error processing the document file.")
        return ""

# LLM pipeline
def llm_pipeline(filepath):
    """Generate summary with error handling and memory management."""
    try:
        # Extract limited text from PDF
        input_text = file_preprocessing(filepath, max_chars=3000)
        
        if not input_text:
            return "Error: Could not extract text from PDF."
        
        # Create summarization pipeline
        pipe_sum = pipeline(
            'summarization',
            model=base_model,
            tokenizer=tokenizer,
            max_length=500,
            min_length=50,
            truncation=True)
        
        # Generate summary
        result = pipe_sum(input_text)
        result = result[0]['summary_text']
        return result
        
    except RuntimeError as e:
        if "memory" in str(e).lower():
            st.error("Memory error: Document is too large. Try a smaller PDF.")
            speak("Memory error: Document is too large. Try a smaller PDF.")
            return "Error: Document too large to summarize. Please try a smaller PDF file."
        else:
            st.error(f"Runtime error during summarization: {str(e)}")
            speak("Error occurred during summarization.")
            return "Error occurred during summarization."
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        speak("Unable to generate summary due to an error.")
        return "Unable to generate summary due to an error."

def answer_query_t5(query, text):
    """Answer queries with error handling and text truncation."""
    try:
        # Limit context size to prevent memory issues
        max_context_chars = 2000
        if len(text) > max_context_chars:
            text = text[:max_context_chars]
        
        prompt = f"question: {query} context: {text}"
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        answer_ids = base_model.generate(inputs, max_length=200, num_return_sequences=1, num_beams=3, early_stopping=True)
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer
    except RuntimeError as e:
        if "memory" in str(e).lower():
            return "Error: Not enough memory to process this query. Try a shorter question or smaller document."
        else:
            return f"Error processing query: {str(e)}"
    except Exception as e:
        return f"Unable to answer query: {str(e)}"

# Function to extract text from the PDF file
def extract_text_from_pdf(file_path):
    """Extract text from PDF with error handling."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        speak("Error extracting text from the PDF file.")
        return ""

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
    """Display PDF with error handling."""
    try:
        # Opening file from file path
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # Embedding PDF in HTML
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

        # Displaying File
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        speak("Error displaying the PDF file.")

# Shreya code
def extract_page_number(command):
    try:
        words = command.split()
        page_index = words.index("page")
        page_number = int(words[page_index + 1])
        return page_number
    except (ValueError, IndexError):
        st.sidebar.warning("Invalid page command.")
        speak("Invalid page command. Please say page followed by a number.")
        return None

def listen_for_command(recognizer):
    while True:
        with sr.Microphone() as source:
            speak("Listening for a command...")
            st.sidebar.info("Listening for a command...")
            audio = recognizer.listen(source, timeout=10)
            try:
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Command: {command}")
                return command
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand audio. Please try again.")
                speak("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                speak(f"Could not request results from Google Speech Recognition service; {e}")

            # If the command is not understood, continue listening for a new command
            continue
def listen_for_command1(recognizer):
    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=1)
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Command: {command}")
                return command
            except WaitTimeoutError:
                # Handle timeout error
                return 'none'
            except UnknownValueError:
                return 'none'
            
def listen_for_question(recognizer):
    while True:
        with sr.Microphone() as source:
            speak("Listening for your question...")
            st.sidebar.info("Listening for your question...")
            audio = recognizer.listen(source, timeout=10)
            try:
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Question: {command}")
                return command
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand audio. Please try again.")
                speak("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                speak(f"Could not request results from Google Speech Recognition service; {e}")

            # If the command is not understood, continue listening for a new command
            continue
            
def read_page(pdf_reader, page_number, speaker, recognizer):
    page = pdf_reader.pages[page_number]
    text = page.extract_text()

    sentences = text.split('.')  # Split text into sentences
    for sentence in sentences:
        speaker.say(sentence)
        speaker.runAndWait()

        # Check if stop command received during reading
        stop_command = listen_for_command1(recognizer)
        if stop_command and "stop" in stop_command:
            st.sidebar.info("Stopping reading...")
            speak("Stopping reading...")
            return  # Exit the function if "stop" command is detected



def read_summarized_content(summary, speaker):
    speaker.say("Reading the summarised content.")
    speaker.say(summary)
    speaker.runAndWait()

# Function to listen for the word to be searched
def listen_for_search_word(recognizer):
    while True:
        with sr.Microphone() as source:
            speak("Please speak the word you want to search for.")
            st.sidebar.info("Please speak the word you want to search for.")
            audio = recognizer.listen(source, timeout=10)
            try:
                search_word = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Search word: {search_word}")
                return search_word
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand the word. Please try again.")
                speak("Could not understand the word. Please try again.")
            except sr.RequestError as e:
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                speak(f"Could not request results from Google Speech Recognition service; {e}")
            # If the word is not understood, continue listening for a new word
            continue

# Function to search for the word in the PDF and return page numbers
def search_word_in_pdf(filepath, search_word):
    """Search for a word in PDF with error handling."""
    try:
        pdf_reader = PyPDF2.PdfReader(filepath)
        total_pages = len(pdf_reader.pages)
        occurrences = []
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text and search_word in text.lower():
                occurrences.append(page_num + 1)  # Adding 1 to convert to 1-based indexing
        return occurrences
    except Exception as e:
        st.error(f"Error searching PDF: {str(e)}")
        speak("Error occurred while searching the PDF.")
        return []

# Helper function for case-insensitive file search
def find_pdf_file(filename, directory="data"):
    """Find PDF file case-insensitively in the given directory."""
    try:
        if not os.path.exists(directory):
            st.error(f"Directory '{directory}' does not exist.")
            speak(f"Directory '{directory}' does not exist. Please create the data folder.")
            return None
        
        # List all files in the directory
        files = os.listdir(directory)
        
        # Search for the file case-insensitively
        for file in files:
            if file.lower() == (filename.lower() + ".pdf"):
                return os.path.join(directory, file)
        
        return None
    except Exception as e:
        st.error(f"Error searching for file: {str(e)}")
        speak("Error occurred while searching for the file.")
        return None

# Streamlit code    
st.set_page_config(layout="wide")

def main():
    st.title("SPEAKLINK : Voice Automated PDF  Assistant for the Visually Impaired")

    recognizer = sr.Recognizer()
    speaker = pyttsx3.init()
    
    greet_user(recognizer)

    querying = False

    # Speech recognition to capture the file name from user's voice
    while True:
        with sr.Microphone() as source:
            st.sidebar.info("Listening for the file name...")
            speak("Listening for the file name...")
            audio = recognizer.listen(source)

            try:
                file_name = recognizer.recognize_google(audio)
                file_name_lower = file_name.lower()
                
                st.sidebar.info(f"File name recognized: {file_name}")

                # Check for exit commands
                if (file_name_lower == "quit" 
                    or "speaklink exit" in file_name_lower
                    or "speak link exit" in file_name_lower
                    or file_name_lower == "exit"):
                    st.sidebar.info("Exiting the program...")
                    speak("Exiting the program...")
                    return

                # Process the file name
                file_name = file_name.replace('underscore', '_').replace(' ', '').strip()
                file_name = file_name.replace('space', ' ')
                
                # Search for file case-insensitively
                filepath = find_pdf_file(file_name, "data")

                # Checking if the file exists
                if filepath:
                    st.sidebar.success(f"File found: {os.path.basename(filepath)}")
                    speak(f"File uploaded successfully")
                    break
                else:
                    st.sidebar.error(f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' or 'exit' to stop.")
                    speak(f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' or 'exit' to stop.")

            except sr.UnknownValueError:
                st.sidebar.error("Sorry, could not understand the audio.")
                speak("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                st.sidebar.error(f"Could not request results from Google Speech Recognition service; {e}")
                speak("Could not request results from speech recognition service.")

    # Reading PDF file
    try:
        pdf_reader = PyPDF2.PdfReader(filepath)
        total_pages = len(pdf_reader.pages)
        current_page = 0
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        speak(f"Error reading PDF file.")
        return

    col1, col2 = st.columns(2)

    with col1:
            st.sidebar.info("Uploaded File")
            displayPDF(filepath)

    with col2:
            while True:
                command = listen_for_command(recognizer)

                if "search" in command:
                    search_word = listen_for_search_word(recognizer)
                    st.sidebar.info(f"Searching for: {search_word}")
                    page_numbers = search_word_in_pdf(filepath, search_word)
                    if page_numbers:
                        st.sidebar.success(f"Word '{search_word}' found on page(s): {', '.join(map(str, page_numbers))}")
                        speak(f"Word '{search_word}' found on page(s): {', '.join(map(str, page_numbers))}")
                    else:
                        st.sidebar.warning(f"Word '{search_word}' not found in the document.")
                        speak(f"Word '{search_word}' not found in the document.")


                elif "query" in command:
                    if base_model is None or tokenizer is None:
                        st.sidebar.warning("Model not loaded. Query feature unavailable.")
                        speak("Model not loaded. Query feature unavailable.")
                        continue
                        
                    if not querying:
                        st.sidebar.info("Starting querying feature. Please ask your question.")
                        speak("Starting querying feature. Please ask your question.")
                        querying = True

                    while querying:
                        question = listen_for_question(recognizer)
                        if "stop" in question:
                            st.sidebar.info("Stopping querying feature.")
                            speak("Stopping querying feature.")
                            querying = False
                            break
                        else:
                            #st.sidebar.info(f"Question: {question}")
                            text = extract_text_from_pdf(filepath)
                            if text:
                                answer = answer_query_t5(question, text)
                                st.success(f"Answer: {answer}")
                                speak(answer)
                            else:
                                st.warning("Could not extract text from PDF.")
                                speak("Could not extract text from PDF.")


                elif "summarize" in command or "summarise" in command:
                    if base_model is None or tokenizer is None:
                        st.sidebar.warning("Model not loaded. Summarization feature unavailable.")
                        speak("Model not loaded. Summarization feature unavailable.")
                        continue
                        
                    st.sidebar.info("Summarizing...")
                    speak("Generating summary. This may take a moment.")
                    summary = llm_pipeline(filepath)
                    st.success(summary)
                    read_summarized_content(summary, speaker)

                elif "read aloud" in command:
                      st.sidebar.info("Reading aloud...")
                      page_num = 0
                      while page_num < len(pdf_reader.pages):
                         from_page = pdf_reader.pages[page_num]
                         text = from_page.extract_text()
        
                         sentences = text.split('.')  # Split text into sentences
                         for sentence in sentences:
                             speaker.say(sentence)
                             speaker.runAndWait()

            # Check if stop command received during reading
                             command = listen_for_command1(recognizer)
                             if command and ("stop" in command):
                                 st.sidebar.info("Stopping reading...")
                                 speak("Stopping reading...")
                                 break

                         else: 
                              page_num+=1 # This block runs if the inner loop didn't break
                              continue
                         break  # This break will exit the outer loop if "stop" command is detected



                elif "start navigation" in command:
                    st.sidebar.info("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                    speak("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")

                    while True:
                        nav_command = listen_for_command(recognizer)

                        if nav_command is not None:
                            if "next page" in nav_command:
                                current_page = min(current_page + 1, total_pages - 1)
                            elif "previous page" in nav_command:
                                current_page = max(current_page - 1, 0)
                            elif "page" in nav_command:
                                page_number = extract_page_number(nav_command)
                                if page_number is not None and 0 <= page_number < total_pages:
                                    current_page = page_number - 1
                                else:
                                    st.sidebar.warning("Invalid page command.")
                                    speak("Invalid page number.")
                                    continue
                            elif "stop navigation" in nav_command:
                                st.sidebar.info("Stopping navigation.")
                                speak("Stopping navigation.")
                                break
                            else:
                                st.sidebar.warning("Invalid navigation command.")
                                speak("Invalid navigation command.")
                                continue

                            read_page(pdf_reader, current_page, speaker, recognizer)
                        else:
                            st.warning("No command received. Please try again.")
                            speak("No command received. Please try again.")
                            continue
                        
                elif "help" in command:
                    st.sidebar.info("Please let me know with which command I can help you!")
                    speak("Please let me know with which command I can help you!")
                    
                    while True:
                        hcommand = listen_for_command(recognizer)

                        if "read" in hcommand:
                            st.sidebar.info("To read aloud the entire file, say 'read aloud'.")
                            speak("To read aloud the entire file, say 'read aloud'.")
                            continue

                        if "read page" in hcommand:
                            st.sidebar.info("To read a specific page, say 'start navigation' followed by the page number.")
                            speak("To read a specific page, say 'start navigation' followed by the page number.")
                            continue

                        if "summarize" in hcommand:
                            st.sidebar.info("To summarize the entire file, say 'summarize'.")
                            speak("To summarize the entire file, say 'summarize'.")
                            continue

                        if "search word" in hcommand:
                            st.sidebar.info("To search for a specific word in the file, say 'speaklink search' followed by the word.")
                            speak("To search for a specific word in the file, say 'speaklink search' followed by the word.")
                            continue

                        if "query" in hcommand:
                            st.sidebar.info("To query the file, say 'query' followed by your question.")
                            speak("To query the file, say 'query' followed by your question.")
                            continue

                        if ("navigation" in hcommand):
                            st.sidebar.info("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                            speak("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                            continue

                        if "exit help" in hcommand or "stop help" in hcommand:
                            st.sidebar.info("Stopping help feature.")
                            speak("Stopping help feature.")
                            break
                        
                        else:
                            st.sidebar.info("Invalid command, please try again")
                            speak("Invalid command, please try again")
                            continue
                            
                       
                elif ("speaklink stop" in command 
                      or "speak link stop" in command
                      or "speaklink exit" in command
                      or "speak link exit" in command
                      or "speaking exit" in command):
                    st.sidebar.info("Stopping assistant.")
                    speak("Stopping assistant.")
                    break
                
                else:
                    st.sidebar.warning("Invalid command.")
                    speak("Invalid command.")
                    continue
                
                
if __name__ == "__main__":
    main()