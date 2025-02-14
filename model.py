import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(text):
    tokens = tokenizer.tokenize(text)  # Tokenizes into subwords
    return tokens

# Load IMDB dataset
imdb_data = pd.read_csv("imdb_reviews.csv")

# Convert sentiment labels to binary
imdb_data['numeric_sentiment'] = imdb_data['sentiment'].apply(lambda x: 1 if x.strip().lower() == "positive" else 0)

# Split into training and test sets
train_imdb_data, test_imdb_data = train_test_split(imdb_data, test_size=0.2, random_state=42)

# Build vocabulary
word_counts = Counter()
for review in train_imdb_data['review']:
    tokens = tokenize(review)
    word_counts.update(tokens)

vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

# Determine max sequence length (95th percentile)
sequence_max_length = int(np.percentile([len(tokenize(text)) for text in train_imdb_data['review']], 95))

# Function to convert text to sequences
def text_to_sequence(text, vocab, max_length=sequence_max_length):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    sequence = sequence[:max_length]
    sequence += [vocab["<PAD>"]] * (max_length - len(sequence))
    return sequence

# Custom dataset class
class IMDBDataset(Dataset):
    def __init__(self, imdb_data, vocab, max_length=sequence_max_length):
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

# Create datasets and data loaders
train_dataset = IMDBDataset(train_imdb_data, vocab)
test_dataset = IMDBDataset(test_imdb_data, vocab)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the Sentiment RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, output_dim=1, dropout=0.5):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(self.dropout(final_hidden))
        return output

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 256
num_layers = 2

# Initialize model
model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced LR for stability

# Move model to GPU if available
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on macOS.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU (No compatible GPU found).")
    return device

device = get_device()
model.to(device)

print(model)

# Training loop
n_epochs = 5  # Increased epochs for better convergence

for epoch in range(n_epochs):
    print(f'\nEpoch {epoch + 1}/{n_epochs}')

    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in train_loader:
        reviews, labels = batch
        reviews, labels = reviews.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(reviews).squeeze(1)  # Remove extra dimension
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Accuracy calculation (apply sigmoid before thresholding)
        preds = torch.sigmoid(predictions) > 0.5
        batch_acc = (preds == labels).float().mean().item()
        epoch_acc += batch_acc

    print(f"Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {epoch_acc/len(train_loader)*100:.2f}%")

# Evaluation
model.eval()
test_acc = 0
with torch.no_grad():
    for batch in test_loader:
        reviews, labels = batch
        reviews, labels = reviews.to(device), labels.to(device)

        predictions = model(reviews).squeeze(1)
        preds = torch.sigmoid(predictions) > 0.5
        test_acc += (preds == labels).float().mean().item()

print(f"Test Accuracy: {test_acc/len(test_loader)*100:.2f}%")

# Save model
torch.save(model, "model.pth")
print("Model saved successfully!")