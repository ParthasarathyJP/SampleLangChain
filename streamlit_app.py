import streamlit as st
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 🪄 UI Setup
st.title("📜 PDF QA Ritual with Hugging Face")
uploaded_file = st.file_uploader("Upload your ceremonial PDF", type="pdf")
query = st.text_input("Ask a question based on the document")

# 🧠 Load and Process PDF
if uploaded_file and query:
    with st.spinner("Summoning documents..."):
        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()

    # 🔮 LLM Setup
    token = st.secrets["huggingface"]["api_token"]
    llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xl",
    huggingfacehub_api_token=token
    )

    # 🧵 Prompt + Chain
    prompt = ChatPromptTemplate.from_template(
        "Use the context below to answer the question.\n\n{context}\n\nQuestion: {input}"
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    # 🗣️ Ask the Question
    with st.spinner("Retrieving scrollworthy insight..."):
        result = retrieval_chain.invoke({"input": query})
        st.success("🦉 Audit Owl Responds:")
        st.write(result["answer"])