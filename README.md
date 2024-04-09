# Question-Answer Chatbot

This project implements a question-answer chatbot using Streamlit and OpenAI's GPT-3.5 model. It allows users to ask questions and receive responses from the chatbot.

## Directory Structure

The project directory is organized as follows:

- `loaders/`: Contains modules for loading documents from PDF files.
- `splitters/`: Contains modules for splitting documents into sentences using spaCy.
- `vectordb/`: Contains modules for creating a vector database from documents.
- `evaluation.py`: Module for evaluating the chatbot's performance.
- `main.py`: Main module for the question-answer chatbot.
- `requirements.txt`: Specifies the required dependencies for the project.
- `README.md`: This file, providing an overview of the project.

## Usage

To run the question-answer chatbot, execute the `main.py` file:

```bash
git clone https://github.com/rsmahashank/question-answer-chatbot.git
cd question-answer-chatbot
pip install -r requirements.txt
streamlit run app.py

Clean __pycache__ from directories
pip install pyclean
Run `pyclean .` from terminal# langchain_rag_example
