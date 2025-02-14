# Sentiment Analysis with LSTM (IMDB Reviews)

## 📌 Project Overview
This project builds a sentiment analysis model using an **LSTM-based neural network** to classify IMDB movie reviews as **positive** or **negative**. The model is trained using PyTorch and utilizes **BERT-based tokenization** for preprocessing.

## 📂 Project Structure
```
sentiment_analysis/
│── data/
│   └── imdb_reviews.csv        # Dataset
│── models/
│   └── sentiment_rnn.py         # LSTM model definition
│── preprocessing.py             # Tokenization & dataset preparation
│── train.py                     # Training script
│── evaluate.py                  # Model evaluation script
│── predict.py                   # Inference script
│── config.py                     # Configuration settings
│── requirements.txt              # Dependencies
│── README.md                     # Project documentation
```

## 🚀 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/erayalp808/rnn-sentiment-analysis-imdb.git
cd rnn-sentiment-analysis-imdb
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Data Preprocessing
- The IMDB dataset (`imdb_reviews.csv`) is loaded using **pandas**.
- Sentiment labels are converted to **binary values** (1 = Positive, 0 = Negative).
- Text is tokenized using **BERT's `AutoTokenizer`**.
- A vocabulary is built based on training data.
- Reviews are **converted to sequences** and padded to a fixed length.
- Data is split into **training (80%)** and **test (20%)** using `train_test_split`.

---

## Model Training
Run the following command to train the model:
```bash
python train.py
```
- Uses **Bidirectional LSTM** with:
  - **128-dimensional embeddings**
  - **256 hidden units**
  - **2 LSTM layers**
  - **Dropout (0.5) for regularization**
- Uses **Binary Cross Entropy Loss** for optimization.
- Runs for **5 epochs** (configurable in `config.py`).
- Saves the trained model to `models/sentiment_model.pth`.

---

## 🎯 Model Evaluation
Run:
```bash
python evaluate.py
```
This script:
- Loads the trained model.
- Computes test accuracy using the test dataset.
- Prints the final accuracy score.

---

## Making Predictions
To predict sentiment for a new review, run:
```bash
python main.py
```
Example usage:
```python
sample_review = "This movie was absolutely fantastic! I loved every moment."
print(predict(sample_review))  # Output: "positive"
```

---

## ⚙️ Configuration
Hyperparameters and file paths are stored in `config.py`:
```python
DATA_PATH = "data/imdb_reviews.csv"
MODEL_PATH = "models/sentiment_model.pth"
BATCH_SIZE = 64
N_EPOCHS = 5
LEARNING_RATE = 0.0005
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
OUTPUT_DIM = 1
DROPOUT = 0.5
```
Modify these values to experiment with different settings.

---

## 🛠 Dependencies
Install all dependencies using:
```bash
pip install -r requirements.txt
```
**Main Libraries Used:**
- `torch` (PyTorch for model training)
- `transformers` (for BERT-based tokenization)
- `pandas` (for data manipulation)
- `sklearn` (for dataset splitting)
- `numpy` (for numerical operations)


