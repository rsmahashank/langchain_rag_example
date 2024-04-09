"""
Module for splitting documents into chunks.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter

chunk_size =1500
chunk_overlap = 150


def split_documents(documents):
    """
    Split documents into chunks.

    Parameters:
        documents (list): List of document objects.

    Returns:
        list: List of chunks.
    """
    rc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    # separators=separators,
    # length_function=len
)
    splits = rc_splitter.split_documents(documents)
    return splits
