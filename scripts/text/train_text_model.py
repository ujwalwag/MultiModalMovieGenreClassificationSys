import re
import nltk
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Text preprocessing functions
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# PyTorch-based tokenizer
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

# Dataset class
class MovieGenreDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM Model
class GenreLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.3):
        super(GenreLSTM, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embedding layer with pretrained weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # freeze embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           batch_first=True,
                           bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to output genre logits
        self.fc = nn.Linear(hidden_dim * 2, 10)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)  # average over time steps
        dropped = self.dropout(pooled)
        output = self.fc(dropped)
        return output

def load_glove_embeddings(glove_path, embedding_dim=100):
    """Load GloVe embeddings"""
    embedding_index = {}
    try:
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = vector
        print(f"✅ Loaded {len(embedding_index)} word vectors from GloVe.")
        return embedding_index
    except FileNotFoundError:
        print(f"❌ GloVe file not found at {glove_path}")
        print("Please download GloVe embeddings from https://nlp.stanford.edu/projects/glove/")
        return {}

def create_embedding_matrix(tokenizer, embedding_index, embedding_dim=100):
    """Create embedding matrix for the tokenizer"""
    vocab_size = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    print(f"✅ Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix

def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    """Train the model"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        all_y_true = []
        all_y_pred = []
        
        for batch in train_loader:
            X_batch, y_batch = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().detach().numpy()
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend((preds >= 0.5).astype(int))
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        train_f1 = f1_score(all_y_true, all_y_pred, average='micro')
        train_f1s.append(train_f1)
        
        # Validation
        val_loss, val_f1 = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        print(f"📚 Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    return train_losses, val_losses, train_f1s, val_f1s

def validate_model(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = [b.to(device) for b in batch]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend((preds >= 0.5).astype(int))
    
    avg_val_loss = running_loss / len(val_loader)
    val_f1 = f1_score(all_y_true, all_y_pred, average='micro')
    return avg_val_loss, val_f1

def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s):
    """Plot training curves"""
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.title("📉 Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # F1 plot
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label="Train F1", marker='o')
    plt.plot(val_f1s, label="Validation F1", marker='o')
    plt.title("📈 F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/text_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 Starting PyTorch-based text model training...")
    
    # Load dataset
    print("📖 Loading dataset...")
    df = pd.read_csv("data/strictly_balanced_top10.csv")
    
    # Clean text
    print("🧹 Cleaning text data...")
    df['clean_overview'] = df['overview'].apply(clean_text_advanced)
    
    # Save cleaned dataset
    df.to_csv("data/strictly_balanced_top10_cleaned.csv", index=False)
    print("✅ Cleaned dataset saved")
    
    # Tokenization
    print("🔤 Tokenizing text...")
    tokenizer = PyTorchTokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_overview'])
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(df['clean_overview'])
    X_text = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    
    print(f"✅ Tokenized text shape: {X_text.shape}")
    
    # Prepare labels
    genre_columns = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                     'Horror', 'Documentary', 'Animation', 'Music', 'Crime']
    y_labels = df[genre_columns].values.astype('float32')
    
    print(f"✅ Label matrix shape: {y_labels.shape}")
    
    # Load GloVe embeddings
    print("🔍 Loading GloVe embeddings...")
    glove_path = "data/glove.6B.100d.txt"
    embedding_index = load_glove_embeddings(glove_path)
    
    if not embedding_index:
        print("⚠️  Using random embeddings (GloVe not available)")
        embedding_matrix = np.random.randn(20000, 100).astype('float32')
    else:
        embedding_matrix = create_embedding_matrix(tokenizer, embedding_index)
    
    # Save processed data
    np.save("X_text.npy", X_text)
    np.save("y_labels.npy", y_labels)
    np.save("models/embedding_matrix.npy", embedding_matrix)
    
    # Save tokenizer
    with open('models/tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("💾 Saved preprocessed data and tokenizer")
    
    # Create dataset and dataloaders
    print("📦 Creating PyTorch datasets...")
    dataset = MovieGenreDataset(X_text, y_labels)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"📦 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Initialize model
    print("🏗️  Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    model = GenreLSTM(embedding_matrix).to(device)
    
    # Train model
    print("🎯 Starting training...")
    train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, epochs=15, device=device
    )
    
    # Plot results
    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s)
    
    # Save model
    print("💾 Saving model...")
    torch.save(model.state_dict(), 'models/genre_classifier.pth')
    print("✅ Model saved successfully!")
    
    # Test prediction
    print("🧪 Testing model...")
    model.eval()
    test_text = "A young boy discovers he has magical powers and goes on an adventure."
    test_clean = clean_text_advanced(test_text)
    test_seq = tokenizer.texts_to_sequences([test_clean])[0][:200]
    test_arr = np.zeros((1, 200), dtype=np.int64)
    test_arr[0, :len(test_seq)] = test_seq
    test_tensor = torch.tensor(test_arr, dtype=torch.long, device=device)
    
    with torch.no_grad():
        test_output = torch.sigmoid(model(test_tensor))[0].cpu().numpy()
    
    top3 = test_output.argsort()[-3:][::-1]
    print(f"🎬 Test prediction for '{test_text}':")
    for i, idx in enumerate(top3):
        print(f"   {i+1}. {genre_columns[idx]}: {test_output[idx]:.3f}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    main()

