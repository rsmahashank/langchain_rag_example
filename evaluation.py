"""
Module for evaluating the performance of the chatbot.
"""

import logging

def evaluate_chatbot(chatbot, evaluation_dataset):
    """
    Evaluate the performance of the chatbot.

    Parameters:
        chatbot (callable): A function that takes a question as input and returns the chatbot's response.
        evaluation_dataset (list): A list of tuples containing (question, ground_truth_answer) pairs for evaluation.

    Returns:
        float: The accuracy of the chatbot on the evaluation dataset.
    """
    correct_predictions = 0
    total_questions = len(evaluation_dataset)

    for question, ground_truth_answer in evaluation_dataset:
        try:
            chatbot_response = chatbot(question)
            if chatbot_response.strip().lower() == ground_truth_answer.strip().lower():
                correct_predictions += 1
        except Exception as e:
            logging.error(f"Error occurred during evaluation: {str(e)}")

    accuracy = correct_predictions / total_questions
    logging.info(f"Chatbot accuracy: {accuracy}")
    return accuracy
