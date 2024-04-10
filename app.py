# app.py
import os
import datetime
import streamlit as st
import logging
import openai
import mimetypes
import shutil
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from splitters.splitters import split_documents
from loaders.loaders import load_documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.DEBUG)

# Set up memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Initialize OpenAI Embeddings
embedding = OpenAIEmbeddings()

# Determine LLM model based on current date
current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"

# Initialize ChatOpenAI model
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Initialize Chroma vector database
persist_directory = 'docs/chroma/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Initialize ConversationalRetrievalChain
conversational_retrieval_qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

# Streamlit app layout
def main():
    """
    Main function to run the question-answer chatbot.
    """
    st.title("Question-Answer Chatbot")

    # Sidebar tab selection
    tab_selection = st.sidebar.selectbox("Select Option", ["Chat", "Upload/Train", "Delete Trained Data"])

    # Chat tab
    if tab_selection == "Chat":
        chat_tab()

    # Upload/Train tab
    elif tab_selection == "Upload/Train":
        train_tab()

    # Delete Trained Data
    elif tab_selection == "Delete Trained Data":
        delete_trained_data_tab()     

# Chat tab
def chat_tab():
    """
    Function to display the chat interface.
    """
    st.title("Chat")

    # Text input for user question
    user_question = st.text_input("Enter your question:")

    # Listen for "Enter" key press
    if user_question.endswith('\n'):
        user_question = user_question[:-1]  # Remove trailing newline
        response = ask_question(user_question)
        st.text_area("Chatbot Response:", response)
        st.text_input("", key="enter_trigger")  # Add empty input field to trigger Enter key event

    # Button to submit question
    if st.button("Ask"):
        if user_question:
            response = ask_question(user_question)
            st.text_area("Chatbot Response:", response)
        else:
            st.warning("Please enter a question.")

# Function to ask question to the conversational QA chain
def ask_question(question):
    """
    Function to ask question to the conversational QA chain.

    Parameters:
        question (str): The question to ask.

    Returns:
        str: The response from the chatbot.
    """
    try:
        result = conversational_retrieval_qa_chain.invoke({"question": question})
        return result["answer"]
    except Exception as e:
        logging.error(f"Error occurred during question answering: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question."

def load_and_preprocess_data(uploaded_files):
    # Load the data from uploaded PDF files
    documents = []
    for uploaded_file in uploaded_files:
        # Get the full file path of the uploaded file
        file_path = os.path.join(os.getcwd(), uploaded_file.name)

        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Use loader to load the PDF file
        loaded_documents = load_documents([file_path])
        documents.extend(loaded_documents)

    # Split the documents
    # splits = split_documents(documents)
    splits = split_documents(documents)
    print(f"# of splits: {len(splits)}")
    return splits


def load_and_preprocess_data1(uploaded_files, chunk_size=1000, chunk_overlap=100):
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            content = uploaded_file.read().decode("utf-8")
            file_type, _ = mimetypes.guess_type(uploaded_file.name)
            documents.append(Document(content, file_type=file_type))

        # Split the documents using the splitter
        splits = split_documents(documents)
        return splits
    else:
        st.warning("Please upload at least one file.")

# Upload/Train tab
def train_tab():
    """
    Function to display the upload/train interface.
    """
    st.title("Upload Files for Training")

    # File uploader
    uploaded_files = st.file_uploader("Upload files for training", type=['pdf', 'csv', 'txt'], accept_multiple_files=True)

    # Training process
    if st.button("Train Model"):
        if uploaded_files:
            try:
                # Load and preprocess data
                splits = load_and_preprocess_data(uploaded_files)
                
                # Create and persist vector database
                vectordb = create_vectordb(splits, embedding, persist_directory)
                vectordb.persist()
                
                st.success("Training completed successfully!")
            except Exception as e:
                logging.error(f"Error occurred during training: {str(e)}")
                st.error("An error occurred during training. Please check the logs for more information.")
        else:
            st.warning("Please upload at least one file.")

def create_vectordb(documents, embedding, persist_directory):
    """
    Create a vector database from a list of documents.

    Parameters:
        documents (list): List of document objects.
        embedding: The embedding model to use for encoding documents.
        persist_directory (str): Directory to persist the vector database.

    Returns:
        VectorDB: The created vector database.
    """
    # Assuming `Chroma` is the VectorDB class
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

def delete_trained_data_tab():
    st.title("Delete Trained Data")

    # Checkbox to confirm deletion
    confirm_delete = st.checkbox("Confirm Training Data Deletion")

    # Button to trigger deletion
    if st.button("Delete Trained Data"):
        delete_trained_data(confirm_delete)

def delete_trained_data(confirm_delete):
    """
    Function to clean the training contents from persist_directory  (vectored store).
    """
    if confirm_delete:
        # Directory containing trained data
        trained_data_dir = 'docs/chroma/'
        
        # Check if the directory exists
        if os.path.exists(trained_data_dir):
            # Delete the directory and its contents
            shutil.rmtree(trained_data_dir)
            
            # Notify the user that the data has been deleted
            st.success("Trained data has been successfully deleted.")
        else:
            st.warning("No trained data found.")
    else:
        st.warning("Please confirm deletion by checking the checkbox.")

if __name__ == "__main__":
    main()
