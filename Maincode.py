import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the model and tokenizer
checkpoint = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# Function to load and split PDF
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return " ".join([t.page_content for t in texts])

# Function to summarize text
def summarize(text):
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# Streamlit UI
st.title("📄 Document Summarization App (CPU Only)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    if st.button("Summarize"):
        with st.spinner("Processing..."):
            text = load_pdf("temp.pdf")
            summary = summarize(text)
            st.subheader("Summary:")
            st.write(summary)