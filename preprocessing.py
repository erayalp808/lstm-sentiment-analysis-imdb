import pandas as pd
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import DATA_PATH, BATCH_SIZE

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text):
    return tokenizer.tokenize(text)

def load_data():
    dataset = load_dataset(DATA_PATH)
    imdb_data = pd.DataFrame(dataset['train'])
    imdb_data['numeric_sentiment'] = imdb_data['sentiment'].apply(lambda x: 1 if x.strip().lower() == "positive" else 0)
    
    train_data, test_data = train_test_split(
        imdb_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=imdb_data['numeric_sentiment']
    )

    return train_data, test_data

def build_vocab(train_data):
    word_counts = Counter()
    for review in train_data['review']:
        word_counts.update(tokenize(review))
    
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def text_to_sequence(text, vocab, max_length):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    sequence = sequence[:max_length] + [vocab["<PAD>"]] * (max_length - len(sequence))
    return sequence

class IMDBDataset(Dataset):
    def __init__(self, imdb_data, vocab, max_length):
        self.reviews = imdb_data['review'].tolist()
        self.labels = imdb_data['numeric_sentiment'].tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx] if isinstance(self.reviews[idx], str) else ""
        label = self.labels[idx]
        sequence = text_to_sequence(review, self.vocab, self.max_length)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def get_data_loaders():
    train_data, test_data = load_data()
    vocab = build_vocab(train_data)

    sequence_max_length = int(np.percentile(
        [len(tokenize(text)) for text in train_data['review']], 95
    ))

    train_dataset = IMDBDataset(train_data, vocab, sequence_max_length)
    test_dataset = IMDBDataset(test_data, vocab, sequence_max_length)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader, vocab, sequence_max_length