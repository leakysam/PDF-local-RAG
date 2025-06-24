import streamlit as st
from newrag import create_vector_db_from_file, process_question

st.title("PDF Question Answering with Local LLM")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question:")

if uploaded_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing document...")
    vector_db = create_vector_db_from_file("temp.pdf")

    st.info("Answering your question...")
    answer = process_question(question, vector_db)

    st.subheader("Answer")
    st.markdown(f"<div style='white-space: pre-wrap;'>{answer}</div>", unsafe_allow_html=True)
