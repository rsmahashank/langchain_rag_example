# loaders/loaders.py
from langchain_core.documents.base import Document
import mimetypes
import csv
from PyPDF2 import PdfReader

def load_documents(file_paths):
    """
    Load documents from files.

    Parameters:
        file_paths (list): List of file paths.

    Returns:
        list: List of loaded documents.
    """
    documents = []
    for file_path in file_paths:
        file_type, _ = mimetypes.guess_type(file_path)
        if file_type == 'text/plain':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(content, file_type=file_type))
        elif file_type == 'text/csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    content = ' '.join(row)  # Join CSV rows into a single string
                    documents.append(Document(content, file_type=file_type))
        elif file_type == 'application/pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    documents.append(Document(content, file_type=file_type))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    return documents