"""
Module for loading documents from PDF files.
"""

from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_paths):
    """
    Load documents from PDF files.

    Parameters:
        file_paths (list): List of file paths to PDF files.

    Returns:
        list: List of loaded documents.
    """
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    return documents
