# Multi-Modal Movie Genre Classification System

A Flask web app for **multi-label movie genre classification** using both **text** (plot) and **visual** (poster) inputs. It combines an LSTM text model with GloVe embeddings and a ResNet-18 image model, trained on a balanced TMDB dataset across the top 10 genres. The stack is **Python 3.8+**, **PyTorch**, and **Flask**.

---

## Live demo

- [Demo 1](https://myapp-284861369113.us-central1.run.app/)
- [Demo 2](https://movie-genre-classification-sys.onrender.com)

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

### Install and run locally

```bash
git clone https://github.com/ujwalwag/Movie-Genre-Classification-Sys.git
cd Movie-Genre-Classification-Sys
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Place the **trained** assets (for example `.pth` weights, `embedding_matrix.npy`, `tokenizer.json`) under `models/` if they are not already present—see `TRAINING_README.md` if you need to train them yourself.

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Enter a plot and/or upload a poster, then **Predict** to see text and image model outputs.

### Run with Docker

```bash
docker build -t movie-genre-classifier .
docker run -p 5000:5000 movie-genre-classifier
```

---

## Training your own models

Training expects dataset files under `data/` (see [TRAINING_README.md](TRAINING_README.md) for filenames and layout).

**End-to-end pipeline** (GloVe download, text + image training, artifacts under `models/` and `plots/`):

```bash
python scripts/train_all_models.py
```

For step-by-step options, per-model scripts, and troubleshooting, see **TRAINING_README.md**. After training, you can run `python test_models.py` to smoke-test the saved weights.

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
