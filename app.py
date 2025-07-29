import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import tempfile

# Function to process file and create vector store
def process_file(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Choose loader based on file type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file type!")
        return None

    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# Create conversational chain
def create_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# Streamlit UI
st.title("Chat with your Document (PDF/Word)")

uploaded_file = st.file_uploader("Upload your file", type=["pdf", "docx"])

if uploaded_file:
    st.write("Processing file... This may take a moment.")
    vectorstore = process_file(uploaded_file)
    chain = create_chain(vectorstore)
    st.success("File processed! You can now chat with it.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about the document:")

    if query:
        result = chain({"question": query})
        answer = result["answer"]

        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))

        for speaker, message in st.session_state.chat_history:
            st.markdown(f"**{speaker}:** {message}")