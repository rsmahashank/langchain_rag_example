"""
Main module for the question-answer chatbot.
"""

import os
import datetime
import streamlit as st
import logging
import openai
from langchain_community.vectorstores import Chroma
from loaders.pdf_loader import load_documents
from splitters.rc_text_splitter import split_documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

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
    tab_selection = st.sidebar.selectbox("Select Option", ["Chat", "Upload/Train"])

    # Chat tab
    if tab_selection == "Chat":
        chat_tab()

    # Upload/Train tab
    elif tab_selection == "Upload/Train":
        train_tab()

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


# Function to load and preprocess data
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

# Upload/Train tab
def train_tab():
    """
    Function to display the upload/train interface.
    """
    st.title("Upload Files for Training")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF documents for training", type=['pdf'], accept_multiple_files=True)

    # Training process
    if st.button("Train Model"):
          if uploaded_files:
            try:
                # Load and preprocess data
                splits = load_and_preprocess_data(uploaded_files)
                
                
                # Create and persist vector database
                vectordb = create_vectordb(splits, embedding, persist_directory)

                vectordb.persist()
                
                st.write("Training completed successfully!")
                
            except Exception as e:
                logging.error(f"Error occurred during training: {str(e)}")
                st.error("An error occurred during training. Please check the logs for more information.")
          else:
            st.warning("Please upload at least one PDF document.")


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


if __name__ == "__main__":
    main()
