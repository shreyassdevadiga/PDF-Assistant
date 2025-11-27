import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64

# -------------------------------
# Model and tokenizer loading
# -------------------------------
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float32
)

# -------------------------------
# File loader and preprocessing
# -------------------------------
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    final_text = ""
    for text in texts:
        final_text += text.page_content
    return final_text

# -------------------------------
# LLM summarization pipeline
# -------------------------------
def llm_pipeline(file_path):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
        device=0 if torch.cuda.is_available() else -1
    )
    input_text = file_preprocessing(file_path)
    result = pipe_sum(input_text)
    return result[0]["summary_text"]

# -------------------------------
# Display PDF in Streamlit
# -------------------------------
@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        filepath = f"data/{uploaded_file.name}"
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)

            with col2:
                st.info("Processing...")
                summary = llm_pipeline(filepath)
                st.success("Summarization Complete")
                st.write(summary)

if __name__ == "__main__":
    main()
