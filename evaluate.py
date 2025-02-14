import torch
from models.sentiment_rnn import SentimentRNN
from preprocessing import get_data_loaders
from config import *

_, test_loader, vocab, _ = get_data_loaders()
model = SentimentRNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

test_acc = 0
with torch.no_grad():
    for reviews, labels in test_loader:
        reviews, labels = reviews.to(DEVICE), labels.to(DEVICE)
        predictions = model(reviews).squeeze(1)
        preds = torch.sigmoid(predictions) > 0.5
        test_acc += (preds == labels).float().mean().item()

print(f"Test Accuracy: {test_acc / len(test_loader) * 100:.2f}%")