# 🎬 Multi-Modal Movie Genre Classification System

A Flask web app for **multi-label movie genre classification** using both **textual (plot)** and **visual (poster image)** inputs. Combines LSTM (text) and ResNet-18 (image) models trained on a balanced TMDB dataset across the top 10 genres.

---

## 🚀 Live Demo

- [Demo 1](https://myapp-284861369113.us-central1.run.app/)
- [Demo 2](https://movie-genre-classification-sys.onrender.com)

---

## 🧠 Features

- **Text-based Genre Classification** (LSTM + GloVe embeddings)
- **Image-based Genre Classification** (ResNet-18)
- **Multi-modal predictions** for accurate multi-label output
- **Web interface**: Enter plot and/or upload poster
- **Visual genre probability predictions**
- **Docker support** for deployment

---

## 🗂️ Project Structure

```
.
├── app.py                # Flask app entry point
├── models/               # Model weights, tokenizer, embedding matrix
├── templates/            # HTML templates (Jinja2)
├── static/               # Static assets (CSS, JS)
├── scripts/              # Training scripts
├── notebook/             # Jupyter notebooks
├── plots/                # Model evaluation plots
├── webapp/               # Web deployment configs
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker instructions
```

---

## 📈 Supported Genres

- Drama
- Comedy
- Romance
- Thriller
- Action
- Horror
- Documentary
- Animation
- Music
- Crime

---

## ⚡ Getting Started

### Prerequisites

- Python 3.8+
- pip
- Git
- torch, torchvision
- Docker (optional, for deployment)

### 💡 How to Use (For People Cloning the Repo)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ujwalwag/MultiModalMovieGenreClassificationSys.git
   cd MultiModalMovieGenreClassificationSys
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights and assets:**
   - Place the required model files (`.pth`, `embedding_matrix.npy`, `tokenizer.json`, etc.) in the `models/` directory.
   - If not included, follow instructions in the repo or contact the maintainer.

5. **Run the Flask app:**
   ```bash
   python app.py
   ```

6. **Open your browser:**
   - Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

7. **Use the web interface:**
   - Enter a movie plot and/or upload a poster image.
   - Click **Predict** to see genre predictions from both text and image models.

---

**Tip:**  
For Docker deployment, use the provided `Dockerfile` and follow the Docker instructions
---

## 🧠 Model Details

### Text Model (LSTM)

- GloVe (100d) embeddings
- Custom tokenizer (`models/tokenizer.json`)
- BiLSTM → Mean Pooling → Dense layers
- Embedding matrix: `models/embedding_matrix.npy`

### Image Model (ResNet-18)

- Pretrained ResNet-18 (torchvision)
- Final FC layer: 10-class sigmoid output
- Poster images normalized and resized

---

## 🤝 Acknowledgements

- TMDB Dataset
- GloVe Embeddings (Stanford)
- PyTorch, Flask, torchvision

---
