# 🎬 MMGCS Training Guide

This guide explains how to train both the text and image models for the Multi-Modal Genre Classification System using PyTorch.

## 🚀 Quick Start

To train all models at once, run:

```bash
python scripts/train_all_models.py
```

This will:
1. Download GloVe embeddings (if needed)
2. Train the text classification model
3. Download movie posters and train the image classification model
4. Save all models and generate training plots

## 📋 Prerequisites

### Required Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```

### Dataset
Ensure you have the dataset file:
- `data/strictly_balanced_top10.csv` - Contains movie data with plots and poster URLs

## 🔤 Text Model Training

### What it does:
- Preprocesses movie plot text (cleaning, tokenization, lemmatization)
- Downloads GloVe word embeddings (400MB)
- Trains a bidirectional LSTM model with attention
- Saves the trained model, tokenizer, and embedding matrix

### Run individually:
```bash
python scripts/text/train_text_model.py
```

### Model Architecture:
- **Embedding Layer**: GloVe 100d vectors (frozen)
- **LSTM**: Bidirectional, 128 hidden units
- **Output**: 10 genre classes with sigmoid activation
- **Loss**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam (lr=0.001)

### Training Details:
- **Epochs**: 15
- **Batch Size**: 32
- **Train/Val Split**: 80/20
- **Data Augmentation**: None (text is already cleaned)

## 🖼️ Image Model Training

### What it does:
- Downloads movie posters from URLs in the dataset
- Applies data augmentation (random crop, flip, color jitter)
- Trains a ResNet18 model with transfer learning
- Saves the best model based on validation F1 score

### Run individually:
```bash
python scripts/image/train_image_model.py
```

### Model Architecture:
- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Transfer Learning**: Freeze early layers, train later layers
- **Custom Head**: Dropout + Linear layers (512 → 10)
- **Output**: 10 genre classes with sigmoid activation
- **Loss**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam (lr=0.001)

### Training Details:
- **Epochs**: 25
- **Batch Size**: 16
- **Train/Val Split**: 80/20
- **Data Augmentation**: Random crop, horizontal flip, color jitter
- **Learning Rate Scheduling**: ReduceLROnPlateau

### Poster Download:
- Downloads posters from `poster_url` column
- Saves as `{movie_id}.jpg` in `data/images/`
- Includes error handling and rate limiting
- Skips already downloaded images

## 📊 Training Monitoring

### Metrics Tracked:
- **Loss**: Training and validation loss per epoch
- **F1 Score**: Micro-averaged F1 score per epoch
- **Learning Rate**: Automatically adjusted based on validation loss

### Plots Generated:
- `plots/text_training_curves.png` - Text model training curves
- `plots/image_training_curves.png` - Image model training curves

## 💾 Model Outputs

### Text Model Files:
- `models/genre_classifier.pth` - Trained LSTM model weights
- `models/tokenizer.pickle` - PyTorch-based tokenizer
- `models/embedding_matrix.npy` - GloVe embedding matrix

### Image Model Files:
- `models/poster_genre_classifier.pth` - Trained ResNet model weights
- `models/poster_genre_classifier_best.pth` - Best model based on validation F1

### Data Files:
- `X_text.npy` - Tokenized and padded text sequences
- `y_labels.npy` - Multi-label genre targets
- `data/images/` - Downloaded movie posters

## 🧪 Testing Models

After training, test both models:

```bash
python test_models.py
```

This will:
- Load both trained models
- Run sample predictions
- Verify model functionality

## ⚠️ Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size in training scripts
   - Use CPU training by setting `device = 'cpu'`

2. **GloVe Download Fails**:
   - Check internet connection
   - Manual download: https://nlp.stanford.edu/projects/glove/
   - Place `glove.6B.100d.txt` in `data/` folder

3. **Poster Download Fails**:
   - Check poster URLs in dataset
   - Verify internet connection
   - Some URLs may be expired or invalid

4. **Model Loading Errors**:
   - Ensure all model files are present
   - Check file permissions
   - Verify PyTorch version compatibility

### Performance Tips:

1. **GPU Training**: Set `device = 'cuda'` for faster training
2. **Batch Size**: Increase if memory allows
3. **Data Augmentation**: Adjust augmentation parameters in image training
4. **Early Stopping**: Monitor validation metrics to prevent overfitting

## 🔧 Customization

### Text Model:
- Modify `hidden_dim` and `dropout` in `GenreLSTM` class
- Adjust vocabulary size in `PyTorchTokenizer`
- Change sequence length in `pad_sequences`

### Image Model:
- Switch to different architectures (ResNet50, EfficientNet)
- Modify data augmentation pipeline
- Adjust learning rate and scheduler parameters

### Training Parameters:
- Modify epochs, batch size, and learning rate
- Add early stopping or model checkpointing
- Implement custom loss functions

## 📈 Expected Results

### Text Model:
- **Training Time**: 10-30 minutes (CPU), 2-5 minutes (GPU)
- **Validation F1**: 0.65-0.80 (depending on data quality)
- **Best Genres**: Drama, Comedy, Action (most common)

### Image Model:
- **Training Time**: 30-60 minutes (CPU), 5-15 minutes (GPU)
- **Validation F1**: 0.55-0.75 (poster quality dependent)
- **Best Genres**: Action, Horror, Animation (visually distinct)

## 🎯 Next Steps

After successful training:

1. **Test the web app**: `python app.py`
2. **Upload movie plots** for text-based classification
3. **Upload movie posters** for image-based classification
4. **Combine predictions** for multi-modal results
5. **Fine-tune models** based on performance analysis

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Multi-Label Classification Guide](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

---

**Happy Training! 🎬🚀**

