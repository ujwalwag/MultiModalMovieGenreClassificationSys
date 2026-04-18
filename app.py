import os
import __main__
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
from torchvision import models, transforms

# ─────────── Config ───────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = torch.device("cpu")
print(f"✅ Using device: {DEVICE}")

# Repo root (works with gunicorn / Render regardless of process cwd)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def _torch_load_weights(path, *, map_location):
    """Load a state_dict .pth; prefer weights_only=True when supported (PyTorch 2.4+)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


app = Flask(__name__, static_folder='static', template_folder='templates')

GENRE_COLUMNS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
    'Horror', 'Documentary', 'Animation', 'Music', 'Crime'
]

IMG_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────── PyTorch Tokenizer ───────────
class PyTorchTokenizer:
    def __init__(self, num_words=20000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
        
    def fit_on_texts(self, texts):
        """Build vocabulary from texts"""
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and limit to num_words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.num_words-1]  # -1 for OOV token
        
        # Build word_index and index_word
        self.word_index[self.oov_token] = 0
        for i, (word, _) in enumerate(sorted_words, 1):
            self.word_index[word] = i
            self.index_word[i] = word
            
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index[self.oov_token])
            sequences.append(sequence)
        return sequences

def pad_sequences(sequences, maxlen=200, padding='post', truncating='post'):
    """Pad sequences to fixed length"""
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) > maxlen:
            if truncating == 'post':
                sequence = sequence[:maxlen]
            else:
                sequence = sequence[-maxlen:]
        else:
            if padding == 'post':
                sequence = sequence + [0] * (maxlen - len(sequence))
            else:
                sequence = [0] * (maxlen - len(sequence)) + sequence
        padded_sequences.append(sequence)
    return np.array(padded_sequences)

# ─────────── Text Model Definition ───────────
class GenreLSTM(nn.Module):
    def __init__(self, emb, hid=128, drop=0.3):
        super().__init__()
        vocab_size, emb_dim = emb.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb, dtype=torch.float32), requires_grad=False)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, len(GENRE_COLUMNS))

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))
        pooled = lstm_out.mean(dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped)


# Tokenizer was pickled while running the training script as __main__, so the
# class path is __main__.PyTorchTokenizer. Under gunicorn, __main__ is not this
# module — register our class there so pickle.load succeeds on Render.
_TEXT_MODEL_CACHE = None


def _get_cached_text_model():
    global _TEXT_MODEL_CACHE
    if _TEXT_MODEL_CACHE is not None:
        return _TEXT_MODEL_CACHE

    __main__.PyTorchTokenizer = PyTorchTokenizer

    tok_path = os.path.join(MODEL_DIR, "tokenizer.pickle")
    emb_path = os.path.join(MODEL_DIR, "embedding_matrix.npy")
    weights_path = os.path.join(MODEL_DIR, "genre_classifier.pth")

    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    embedding_matrix = np.load(emb_path)
    txt_state = _torch_load_weights(weights_path, map_location=DEVICE)

    model = GenreLSTM(embedding_matrix).to(DEVICE)
    model.load_state_dict(txt_state)
    model.eval()

    _TEXT_MODEL_CACHE = (tokenizer, model)
    return _TEXT_MODEL_CACHE


# ─────────── Routes ───────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json(silent=True) or {}
    plot = (data.get("plot") or data.get("text") or "").strip()
    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    try:
        tokenizer, model = _get_cached_text_model()

        print("🔍 Tokenizing input...")
        seq = tokenizer.texts_to_sequences([plot])[0]
        arr = pad_sequences([seq], maxlen=200, padding='post', truncating='post')
        tensor = torch.tensor(arr, dtype=torch.long, device=DEVICE)

        print("🔍 Predicting...")
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].cpu().numpy()

        top3 = probs.argsort()[-3:][::-1]
        print("✅ Prediction done.")
        return jsonify({"genres": [GENRE_COLUMNS[i] for i in top3]})

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return jsonify({"error": f"Failed to predict: {str(e)}"}), 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if 'poster' not in request.files:
        return jsonify({"error": "No poster uploaded"}), 400

    try:
        img = Image.open(request.files['poster'].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(GENRE_COLUMNS))
        state = _torch_load_weights(
            os.path.join(MODEL_DIR, "poster_genre_classifier.pth"),
            map_location=DEVICE,
        )
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()

        tensor = IMG_TF(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].cpu().numpy()
        top3 = probs.argsort()[-3:][::-1]
        return jsonify({"genres": [GENRE_COLUMNS[i] for i in top3]})

    except Exception as e:
        return jsonify({"error": f"Failed to predict image genre: {str(e)}"}), 500

# ─────────── Launch ───────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

