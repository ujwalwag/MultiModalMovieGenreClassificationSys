# NeuralNetflix - Multi-Modal Movie Genre Classification System

A Flask web app for **multi-label movie genre classification** using both **textual (plot)** and **visual (poster image)** inputs. Combines LSTM (text) and ResNet-18 (image) models trained on a balanced TMDB dataset across the top 10 genres.

---

## Live demo

- [Demo 1](https://myapp-284861369113.us-central1.run.app/)
- [Demo 2](https://multimodalmoviegenreclassificationsys.onrender.com)

---

## Features

- Text-based genre classification (LSTM + GloVe)
- Image-based genre classification (ResNet-18)
- Multi-label outputs (top genres per modality from the API)
- Web UI: enter a plot and/or upload a poster

---

## Project structure

```
.
├── app.py                 # Flask app entry point
├── models/                # Model weights, tokenizer pickle, embedding matrix
├── data/                  # Datasets for training (gitignored — add locally)
├── templates/             # HTML (Jinja2)
├── static/                # Assets (images, optional sample data)
├── scripts/               # Training & data prep (see TRAINING_README.md)
├── notebook/              # Jupyter notebooks
├── plots/                 # Training / evaluation plots
├── webapp/                # Extra web assets (e.g. JS)
├── TRAINING_README.md     # Full training guide
└── requirements.txt
```

---

## Supported genres

Drama, Comedy, Romance, Thriller, Action, Horror, Documentary, Animation, Music, Crime

---

## Getting started

### Prerequisites

- Python 3.8+
- pip, Git
- PyTorch / torchvision (via `requirements.txt`)

### Installation

```bash
git clone https://github.com/ujwalwag/Movie-Genre-Classification-Sys.git
cd Movie-Genre-Classification-Sys
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Model files

Place trained assets under `models/` (paths are fixed in `app.py`):

| File | Role |
|------|------|
| `genre_classifier.pth` | Text model state dict |
| `tokenizer.pickle` | Vocabulary / tokenizer (pickle) |
| `embedding_matrix.npy` | Frozen GloVe embedding table |
| `poster_genre_classifier.pth` | Image (ResNet-18) state dict |

If these are missing, train or copy them in (see **TRAINING_README.md**).

### Run locally

```bash
python app.py
```

Default port is **`10000`** unless you set the **`PORT`** environment variable (e.g. `5000` on Windows: `$env:PORT=5000; python app.py`).

Open [http://127.0.0.1:10000](http://127.0.0.1:10000) (or your chosen `PORT`).

### Training

End-to-end training (see **TRAINING_README.md** for data paths and options):

```bash
python scripts/train_all_models.py
```

---

## Model details

### Text (LSTM)

- GloVe 100d embeddings (frozen in `embedding_matrix.npy`)
- Tokenizer: `models/tokenizer.pickle` (not JSON)
- BiLSTM, mean pooling, dense head → `genre_classifier.pth`

### Image (ResNet-18)

- torchvision ResNet-18 (ImageNet weights only used during training; inference loads saved weights)
- Final layer: 10 outputs with sigmoid for multi-label poster classification
- Input: posters resized/cropped to 224×224 and ImageNet-normalized

---

## Acknowledgements

- TMDB dataset
- GloVe embeddings (Stanford NLP)
- PyTorch, Flask, torchvision
