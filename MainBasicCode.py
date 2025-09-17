import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os

# Load model and tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    input_text = file_preprocessing(filepath)
    pipe_sum = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    result = pipe_sum(input_text)
    return result[0]['summary_text']

# Cache PDF viewer
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            os.makedirs("data", exist_ok=True)
            filepath = os.path.join("data", uploaded_file.name)
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)

            with col2:
                with st.spinner("Summarizing..."):
                    summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()
