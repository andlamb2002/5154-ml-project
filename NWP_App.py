# Imports

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# BiLSTM Model

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
# Load model and metadata
    
with open("nwp_model_meta.pkl", "rb") as f:
    metadata = pickle.load(f)

word2idx = metadata["word2idx"]
idx2word = metadata["idx2word"]
vocab_size = metadata["vocab_size"]
embed_size = metadata["embed_size"]
hidden_size = metadata["hidden_size"]
num_layers = metadata["num_layers"]

model = BiLSTMModel(
    vocab_size=metadata["vocab_size"],
    embed_size=metadata["embed_size"],
    hidden_size=metadata["hidden_size"],
    num_layers=metadata["num_layers"]
)

model.load_state_dict(torch.load("nwp_model.pth", map_location=torch.device("cpu")))
model.eval()

# NWP App

st.title("Next Word Predictor")

top_k = st.slider("Top K Predictions", min_value=1, max_value=10, value=5)
input_text = st.text_input("Enter a phrase:", "")

if input_text.strip():
    words = input_text.strip().lower().split()
    words = words[-5:] if len(words) > 5 else words
    input_ids = [word2idx.get(w, 0) for w in words]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    input_tensor = torch.nn.functional.pad(input_tensor, (0, 5 - len(input_ids)), value=0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, top_k)

    for i in range(top_k):
        word = idx2word[top_idxs[0][i].item()]
        prob = top_probs[0][i].item()
        st.write(f"{i+1}. {word} ({prob:.2f}%)")
