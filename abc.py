import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
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

# Model and tokenizer loading
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

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
        return None

def listen_for_command(recognizer):
    while True:
        with sr.Microphone() as source:
            pyttsx3.speak("Listening for a command...")
            st.sidebar.info("Listening for a command...")
            audio = recognizer.listen(source, timeout=10)
            try:
                command = recognizer.recognize_google(audio).lower()
                st.sidebar.info(f"Command: {command}")
                return command
            except sr.UnknownValueError:
                st.sidebar.warning("Could not understand audio. Please try again.")
                pyttsx3.speak("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                pyttsx3.speak(f"Could not request results from Google Speech Recognition service; {e}")

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
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                pyttsx3.speak(f"Could not request results from Google Speech Recognition service; {e}")

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
            pyttsx3.speak("Stopping reading...")
            return  # Exit the function if "stop" command is detected



def read_summarized_content(summary, speaker):
    speaker.say("Reading the summarised content.")
    speaker.say(summary)
    speaker.runAndWait()

# Function to listen for the word to be searched
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
                st.sidebar.warning(f"Could not request results from Google Speech Recognition service; {e}")
                pyttsx3.speak(f"Could not request results from Google Speech Recognition service; {e}")
            # If the word is not understood, continue listening for a new word
            continue

# Function to search for the word in the PDF and return page numbers
def search_word_in_pdf(filepath, search_word):
    pdf_reader = PyPDF2.PdfReader(filepath)
    total_pages = len(pdf_reader.pages)
    occurrences = []
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text().lower()
        if search_word in text:
            occurrences.append(page_num + 1)  # Adding 1 to convert to 1-based indexing
    return occurrences


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
            pyttsx3.speak("Listening for the file name...")
            audio = recognizer.listen(source)

            try:
                file_name = recognizer.recognize_google(audio)
                file_name = file_name.replace('underscore', '_').replace(' ', '').strip()
                file_name = file_name.replace('space', ' ')
                st.sidebar.info(f"File name recognized: {file_name}")

                if file_name.lower() == "quit":
                    st.sidebar.info("Exiting the program...")
                    pyttsx3.speak("Exiting the program...")
                    return

                # Appending ".pdf" to the file name
                filepath = os.path.join("data", file_name.strip() + ".pdf")

                # Checking if the file exists
                if os.path.exists(filepath):
                    pyttsx3.speak(f"File uploaded sucessfully")
                    break
                else:
                    st.sidebar.error(f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' to exit.")
                    pyttsx3.speak(f"File '{file_name}.pdf' does not exist. Please try again or say 'quit' to exit.")

            except sr.UnknownValueError:
                st.sidebar.error("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                st.sidebar.error(f"Could not request results from Google Speech Recognition service; {e}")

    # Reading PDF file
    pdf_reader = PyPDF2.PdfReader(filepath)
    total_pages = len(pdf_reader.pages)
    current_page = 0

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
                        pyttsx3.speak(f"Word '{search_word}' found on page(s): {', '.join(map(str, page_numbers))}")
                    else:
                        st.sidebar.warning(f"Word '{search_word}' not found in the document.")
                        pyttsx3.speak(f"Word '{search_word}' not found in the document.")


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
                            #st.sidebar.info(f"Question: {question}")
                            text = extract_text_from_pdf(filepath)
                            answer = answer_query_t5(question, text)
                            st.success(f"Answer: {answer}")
                            pyttsx3.speak(answer)


                elif "summarize" in command or "summarise" in command:
                    st.sidebar.info("Summarizing...")
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
                                 pyttsx3.speak("Stopping reading...")
                                 break

                         else: 
                              page_num+=1 # This block runs if the inner loop didn't break
                              continue
                         break  # This break will exit the outer loop if "stop" command is detected



                elif "start navigation" in command:
                    st.sidebar.info("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                    pyttsx3.speak("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")

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
                                    pyttsx3.speak("Invalid page number.")
                                    continue
                            elif "stop navigation" in nav_command:
                                st.sidebar.info("Stopping navigation.")
                                pyttsx3.speak("Stopping navigation.")
                                break
                            else:
                                st.sidebar.warning("Invalid navigation command.")
                                pyttsx3.speak("Invalid navigation command.")
                                continue

                            read_page(pdf_reader, current_page, speaker, recognizer)
                        else:
                            st.warning("No command received. Please try again.")
                            pyttsx3.speak("No command received. Please try again.")
                            continue
                        
                elif "help" in command:
                    st.sidebar.info("Please let me know with which command I can help you!")
                    pyttsx3.speak("Please let me know with which command I can help you!")
                    
                    while True:
                        hcommand = listen_for_command(recognizer)

                        if "read" in hcommand:
                            st.sidebar.info("To read aloud the entire file, say 'read aloud'.")
                            pyttsx3.speak("To read aloud the entire file, say 'read aloud'.")
                            continue

                        if "read page" in hcommand:
                            st.sidebar.info("To read a specific page, say 'start navigation' followed by the page number.")
                            pyttsx3.speak("To read a specific page, say 'start navigation' followed by the page number.")
                            continue

                        if "summarize" in hcommand:
                            st.sidebar.info("To summarize the entire file, say 'summarize'.")
                            pyttsx3.speak("To summarize the entire file, say 'summarize'.")
                            continue

                        if "search word" in hcommand:
                            st.sidebar.info("To search for a specific word in the file, say 'speaklink search' followed by the word.")
                            pyttsx3.speak("To search for a specific word in the file, say 'speaklink search' followed by the word.")
                            continue

                        if "query" in hcommand:
                            st.sidebar.info("To query the file, say 'query' followed by your question.")
                            pyttsx3.speak("To query the file, say 'query' followed by your question.")
                            continue

                        if ("navigation" in hcommand):
                            st.sidebar.info("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                            pyttsx3.speak("Starting navigation. You can say 'Next Page', 'Previous Page', or 'Page X'.")
                            continue

                        if ("exit help" or "stop help" in hcommand):
                            st.sidebar.info("Stopping help feature.")
                            pyttsx3.speak("Stopping help feature.")
                            break
                        
                        else:
                            st.sidebar.info("Invalid command, please try again")
                            pyttsx3.speak("Invalid command, please try again")
                            continue
                            
                       
                elif ("stop" or "speak link stop"
                or "speaklink stop"
                or "speaking stop"
                or "speaklink exit"
                or "speak link exit"
                or "speaking exit" in command):
                    st.sidebar.info("Stopping assistant.")
                    pyttsx3.speak("Stopping assistant.")
                    break
                
                else:
                    st.sidebar.warning("Invalid command.")
                    pyttsx3.speak("Invalid command.")
                    continue
                
                
if __name__ == "__main__":
    main()

