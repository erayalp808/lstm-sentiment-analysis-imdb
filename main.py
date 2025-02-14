import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sentiment_rnn import SentimentRNN
from preprocessing import get_data_loaders, text_to_sequence
from config import *

def load_model(model_path: str):
    """ Loads and returns model from given path """
    _, _, vocab, sequence_max_length = get_data_loaders()
    model = SentimentRNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, vocab, sequence_max_length

def predict(sample) -> str:
    """ 
    Input: sample (str): single sample containing a movie review.
    Returns: the sentiment type: ["positive", "negative"]
    """
    model, vocab, sequence_max_length = load_model(MODEL_PATH)
    sequence = text_to_sequence(sample, vocab, sequence_max_length)
    input_tensor = torch.tensor([sequence], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor).squeeze(1)
        prediction = torch.sigmoid(output).item()

    return "positive" if prediction > 0.5 else "negative"

"""
# this is a pseudo test function to give an idea, you can delete it if you want
def test_model(test_sample: str, label: str) -> None:
    result = predict(test_sample)
    return result == label
"""

if __name__ == "__main__":
    sample_review = "This movie was absolutely fantastic! I loved every moment."
    print(f"Review: { sample_review }\nPredicted Sentiment: { predict(sample_review) }")
    sample_review = 'I Hated it! absolute mess.'
    print(f"Review: { sample_review }\nPredicted Sentiment: { predict(sample_review) }")