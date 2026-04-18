# Multi-Modal Movie Genre Classification System

A Flask web app for **multi-label movie genre classification** using both **textual (plot)** and **visual (poster image)** inputs. Combines LSTM (text) and ResNet-18 (image) models trained on a balanced TMDB dataset across the top 10 genres.

---

## Live demo

- [Demo 1](https://myapp-284861369113.us-central1.run.app/)
- [Demo 2](https://multimodalmoviegenreclassificationsys.onrender.com)

---

## Features

- Text-based genre classification (LSTM + GloVe)
- Image-based genre classification (ResNet-18)
- Multi-modal predictions for multi-label output
- Web UI: enter a plot and/or upload a poster
- Genre probability visualization
- Docker support for deployment

---

## Project structure

```
.
├── app.py                 # Flask app entry point
├── models/                # Model weights, tokenizer, embedding matrix
├── data/                  # Datasets & downloads for training (not in git — add locally)
├── templates/             # HTML (Jinja2)
├── static/                # CSS, JS, sample assets
├── scripts/               # Training & data prep (see TRAINING_README.md)
├── notebook/              # Jupyter notebooks
├── plots/                 # Training / evaluation plots
├── webapp/                # Deployment configs
├── TRAINING_README.md     # Full training guide
├── requirements.txt
└── Dockerfile
```

---

## Supported genres

Drama, Comedy, Romance, Thriller, Action, Horror, Documentary, Animation, Music, Crime

---

## Getting started

### Prerequisites

- Python 3.8+
- pip, Git
- PyTorch / torchvision (installed via `requirements.txt`)
- Docker (optional, for containerized runs)

### Installation

```bash
git clone https://github.com/ujwalwag/Movie-Genre-Classification-Sys.git
cd Movie-Genre-Classification-Sys
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Run with Docker

```bash
docker build -t movie-genre-classifier .
docker run -p 5000:5000 movie-genre-classifier
```

---
### 💡 How to Use (For People Cloning the Repo)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ujwalwag/Movie-Genre-Classification-Sys.git
   cd Movie-Genre-Classification-Sys
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

## Model details

### Text (LSTM)

- GloVe 100d embeddings
- Custom tokenizer (`models/tokenizer.json`)
- BiLSTM, mean pooling, dense head
- Embedding matrix: `models/embedding_matrix.npy`

### Image (ResNet-18)

- torchvision ResNet-18 backbone (ImageNet pretrained)
- Final layer: 10-class sigmoid for multi-label output
- Posters resized and normalized to match training

---

## Acknowledgements

- TMDB dataset
- GloVe embeddings (Stanford NLP)
- PyTorch, Flask, torchvision
