import torch
import torch.optim as optim
import torch.nn as nn
from models.sentiment_rnn import SentimentRNN
from preprocessing import get_data_loaders
from config import *
import os

train_loader, test_loader, vocab, _ = get_data_loaders()

model = SentimentRNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(N_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
    model.train()
    epoch_loss, epoch_acc = 0, 0

    for reviews, labels in train_loader:
        reviews, labels = reviews.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(reviews).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.sigmoid(predictions) > 0.5
        epoch_acc += (preds == labels).float().mean().item()

    print(f"Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_acc / len(train_loader) * 100:.2f}%")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully!")