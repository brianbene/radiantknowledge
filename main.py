!pip install --upgrade --no-cache-dir PyPDF2 gensim "numpy>=1.26.0" "transformers[torch]" -q
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import json
import gc

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import gensim.downloader as api
except ImportError:
    api = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
#----Mounting google drive-----

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except ImportError:
    print("Not running in Google Colab.")

#Utility funcs to tilize throughout the script

def load_data_from_directory(path):
    texts, labels = [], []
    if not os.path.exists(path): return None, None
    top_level_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for category in top_level_dirs:
        category_path = os.path.join(path, category)
        for root, _, files in os.walk(category_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                text = ""
                try:
                    if filename.endswith(".txt"):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                    elif filename.endswith(".pdf") and PyPDF2:
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text: text += page_text
                    if text:
                        texts.append(text)
                        labels.append(category)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
    print(f"Found {len(texts)} documents.")
    return texts, labels

def train_model(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, iterator):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions = model(text)
            _, predicted_labels = torch.max(predictions, 1)
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

DRIVE_PATH = "/content/drive/MyDrive/"
corpus_path = os.path.join(DRIVE_PATH, "Colab Notebooks/AIPI 540/MylittleRickover/nuclear_corpus/wikipedia")
texts, labels = load_data_from_directory(corpus_path)

#----Needed to split the training data based on text and class labels. input features are text with target as class----
if texts and labels:
    df = pd.DataFrame({'text': texts, 'category': labels})
    X = df['text']
    y = df['category']
    class_labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#---Mean baseline model with dummyclassifier----
    naive_model = DummyClassifier(strategy="most_frequent")
    naive_model.fit(X_train, y_train)
    y_pred_naive = naive_model.predict(X_test)
    accuracy_naive = accuracy_score(y_test, y_pred_naive)
    print(f"Baseline Model Accuracy: {accuracy_naive:.4f}")

#used a NB approach using tfidf scores---

    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Model Accuracy: {accuracy_nb:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    y_train_idx = torch.tensor([label_to_idx[label] for label in y_train], dtype=torch.long)
    y_test_idx = torch.tensor([label_to_idx[label] for label in y_test], dtype=torch.long)

    word_counts = Counter(word for text in X_train for word in text.lower().split())
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)[:5000]
    word_to_idx = {word: i+2 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    word_to_idx['<pad>'] = 1
    vocab_size = len(word_to_idx)
    PAD_IDX = word_to_idx['<pad>']

    def encode_text(text_series):
        return [torch.tensor([word_to_idx.get(word, 0) for word in text.lower().split()]) for text in text_series]

    X_train_enc = encode_text(X_train)
    X_test_enc = encode_text(X_test)
    X_train_pad = pad_sequence(X_train_enc, batch_first=True, padding_value=PAD_IDX)
    X_test_pad = pad_sequence(X_test_enc, batch_first=True, padding_value=PAD_IDX)

    class TextDataset(Dataset):
        def __init__(self, features, labels):
            self.features, self.labels = features, labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    train_dataset = TextDataset(X_train_pad, y_train_idx)
    test_dataset = TextDataset(X_test_pad, y_test_idx)

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class SimpleRNN_PyTorch(nn.Module):
        def __init__(self, v_s, e_d, h_d, o_d, p_i):
            super().__init__()
            self.embedding = nn.Embedding(v_s, e_d, padding_idx=p_i)
            self.rnn = nn.RNN(e_d, h_d, batch_first=True)
            self.fc = nn.Linear(h_d, o_d)
        def forward(self, text):
            _, hidden = self.rnn(self.embedding(text))
            return self.fc(hidden.squeeze(0))

    rnn_model = SimpleRNN_PyTorch(vocab_size, 100, 128, len(class_labels), PAD_IDX).to(device)
    optimizer_rnn = torch.optim.Adam(rnn_model.parameters())
    criterion_rnn = nn.CrossEntropyLoss().to(device)

    for epoch in range(10):
        train_model(rnn_model, train_loader, optimizer_rnn, criterion_rnn)
    accuracy_rnn = evaluate_model(rnn_model, test_loader)
    print(f"Simple RNN Model Accuracy: {accuracy_rnn:.4f}")
    del rnn_model, optimizer_rnn, criterion_rnn
    gc.collect()
    torch.cuda.empty_cache()

    class LSTMClassifier(nn.Module):
        def __init__(self, v_s, e_d, h_d, o_d, n_l, b, d_o, p_i, weights=None):
            super().__init__()
            self.embedding = nn.Embedding(v_s, e_d, padding_idx=p_i)
            if weights is not None:
                self.embedding.weight.data.copy_(weights)
                self.embedding.weight.requires_grad = False
            self.lstm = nn.LSTM(e_d, h_d, num_layers=n_l, bidirectional=b, dropout=d_o, batch_first=True)
            self.fc = nn.Linear(h_d * 2 if b else h_d, o_d)
            self.dropout = nn.Dropout(d_o)
        def forward(self, text):
            embedded = self.embedding(text)
            if self.embedding.weight.requires_grad:
                embedded = self.dropout(embedded)
            _, (hidden, _) = self.lstm(embedded)
            h = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) if self.lstm.bidirectional else self.dropout(hidden[-1,:,:])
            return self.fc(h)

    HIDDEN_DIM_LSTM = 128
    lstm_model = LSTMClassifier(vocab_size, 100, HIDDEN_DIM_LSTM, len(class_labels), 2, True, 0.5, PAD_IDX).to(device)
    optimizer_lstm = torch.optim.Adam(lstm_model.parameters())
    criterion_lstm = nn.CrossEntropyLoss().to(device)

    for epoch in range(10):
        train_model(lstm_model, train_loader, optimizer_lstm, criterion_lstm)
    accuracy_lstm = evaluate_model(lstm_model, test_loader)
    print(f"Bi-LSTM Model Accuracy: {accuracy_lstm:.4f}")
    del lstm_model, optimizer_lstm, criterion_lstm
    gc.collect()
    torch.cuda.empty_cache()

    if api:
        word2vec_model = api.load("word2vec-google-news-300")
        EMBEDDING_DIM_W2V = word2vec_model.vector_size

        embedding_matrix = torch.zeros((vocab_size, EMBEDDING_DIM_W2V))
        for word, i in word_to_idx.items():
            if word in word2vec_model:
                embedding_matrix[i] = torch.FloatTensor(word2vec_model[word])

        embedding_matrix = embedding_matrix.to(device)

        del word2vec_model
        gc.collect()
        torch.cuda.empty_cache()

        lstm_model_w2v = LSTMClassifier(vocab_size, EMBEDDING_DIM_W2V, HIDDEN_DIM_LSTM, len(class_labels), 2, True, 0.5, PAD_IDX, weights=embedding_matrix).to(device)
        optimizer_w2v = torch.optim.Adam(lstm_model_w2v.parameters())
        criterion_w2v = nn.CrossEntropyLoss().to(device)

        for epoch in range(10):
            train_model(lstm_model_w2v, train_loader, optimizer_w2v, criterion_w2v)
        accuracy_w2v = evaluate_model(lstm_model_w2v, test_loader)
        print(f"Bi-LSTM w/Word2Vec Model Accuracy: {accuracy_w2v:.4f}")
        del lstm_model_w2v, optimizer_w2v, criterion_w2v
        gc.collect()
        torch.cuda.empty_cache()
    else:
        accuracy_w2v = 0.0

    MODEL_NAME = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    class BertDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            label = self.labels[item]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    MAX_LEN = 256
    BERT_BATCH_SIZE = 8

    train_bert_dataset = BertDataset(
        texts=X_train.to_numpy(),
        labels=[label_to_idx[l] for l in y_train],
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_bert_dataset = BertDataset(
        texts=X_test.to_numpy(),
        labels=[label_to_idx[l] for l in y_test],
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_bert_loader = DataLoader(train_bert_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)
    test_bert_loader = DataLoader(test_bert_dataset, batch_size=BERT_BATCH_SIZE)

    bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(class_labels))
    bert_model = bert_model.to(device)

    optimizer_bert = AdamW(bert_model.parameters(), lr=2e-5)

    for epoch in range(9):
        print(f"BERT Epoch {epoch + 1}/9")
        bert_model.train()
        for d in train_bert_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer_bert.step()
            optimizer_bert.zero_grad()

    bert_model.eval()
    all_preds_bert, all_labels_bert = [], []
    with torch.no_grad():
        for d in test_bert_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            all_preds_bert.extend(preds.cpu().numpy())
            all_labels_bert.extend(labels.cpu().numpy())

    accuracy_bert = accuracy_score(all_labels_bert, all_preds_bert)
    print(f"BERT Model Accuracy: {accuracy_bert:.4f}")

    ARTIFACTS_PATH = os.path.join(DRIVE_PATH, "NuclearAppArtifacts")
    BERT_MODEL_SAVE_PATH = os.path.join(ARTIFACTS_PATH, "BERT_model_final")
    os.makedirs(BERT_MODEL_SAVE_PATH, exist_ok=True)

    bert_model.save_pretrained(BERT_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(BERT_MODEL_SAVE_PATH)

    LABELS_SAVE_PATH = os.path.join(ARTIFACTS_PATH, 'class_labels.json')
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_labels, f)

    print(f"1. Baseline Accuracy: {accuracy_naive:.4f}")
    print(f"2. Naive Bayes Accuracy: {accuracy_nb:.4f}")
    print(f"3. Simple RNN Accuracy: {accuracy_rnn:.4f}")
    print(f"4. Bi-LSTM (from scratch) Accuracy: {accuracy_lstm:.4f}")
    print(f"5. Bi-LSTM w/Word2Vec Accuracy: {accuracy_w2v:.4f}")
    print(f"6. BERT (fine-tuned) Accuracy: {accuracy_bert:.4f}")

else:
    print("Script finished without running models due to data loading issues.")