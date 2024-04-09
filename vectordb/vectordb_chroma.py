"""
Module for creating a vector database from documents.
"""

from langchain_community.vectorstores import Chroma

def create_vectordb(documents, embedding, persist_directory):
    """
    Create a vector database from documents.

    Parameters:
        documents (list): List of documents.
        embedding: Embedding model.
        persist_directory (str): Directory to persist vector database.

    Returns:
        Chroma: Vector database object.
    """
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        index_options={"dimension": 768, "method": "hnsw", "num_threads": -1}
    )
    vectordb.persist()
    return vectordb
