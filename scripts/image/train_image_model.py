import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Image transformations
IMG_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

IMG_TF_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class MoviePosterDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # Genre columns
        self.genre_columns = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                             'Horror', 'Documentary', 'Animation', 'Music', 'Crime']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Create a black image if poster not found
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = row[self.genre_columns].values.astype('float32')
        
        return image, torch.tensor(labels, dtype=torch.float32)

def download_poster(poster_url, save_path, timeout=10):
    """Download a movie poster from URL"""
    try:
        response = requests.get(poster_url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"❌ Failed to download {poster_url}: {str(e)}")
        return False

def download_all_posters(df, img_dir, max_workers=4):
    """Download all movie posters"""
    print(f"📥 Downloading {len(df)} movie posters...")
    
    # Create image directory
    os.makedirs(img_dir, exist_ok=True)
    
    # Filter rows with poster URLs
    df_with_posters = df[df['poster_url'].notna() & (df['poster_url'] != '')]
    print(f"📊 Found {len(df_with_posters)} movies with poster URLs")
    
    downloaded = 0
    failed = 0
    
    for idx, row in tqdm(df_with_posters.iterrows(), total=len(df_with_posters), desc="Downloading posters"):
        poster_url = row['poster_url']
        movie_id = row['id']
        save_path = os.path.join(img_dir, f"{movie_id}.jpg")
        
        # Skip if already downloaded
        if os.path.exists(save_path):
            downloaded += 1
            continue
            
        if download_poster(poster_url, save_path):
            downloaded += 1
        else:
            failed += 1
        
        # Small delay to be respectful to servers
        time.sleep(0.1)
    
    print(f"✅ Downloaded: {downloaded} | Failed: {failed}")
    return downloaded, failed

def create_resnet_model(num_classes=10, pretrained=True):
    """Create a ResNet model for genre classification"""
    if pretrained:
        model = models.resnet18(pretrained=True)
        print("✅ Using pretrained ResNet18")
    else:
        model = models.resnet18(pretrained=False)
        print("✅ Using ResNet18 from scratch")
    
    # Freeze early layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

def train_model(model, train_loader, val_loader, epochs=20, device='cpu'):
    """Train the image classification model"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0
        all_y_true = []
        all_y_pred = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().detach().numpy()
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend((preds >= 0.5).astype(int))
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        train_f1 = f1_score(all_y_true, all_y_pred, average='micro')
        train_f1s.append(train_f1)
        
        # Validation
        val_loss, val_f1 = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"📚 Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'models/poster_genre_classifier_best.pth')
            print(f"💾 New best model saved! Val F1: {val_f1:.4f}")
    
    return train_losses, val_losses, train_f1s, val_f1s

def validate_model(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_y_true.extend(labels.cpu().numpy())
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
    plt.savefig('plots/image_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 Starting PyTorch-based image model training...")
    
    # Load dataset
    print("📖 Loading dataset...")
    df = pd.read_csv("data/strictly_balanced_top10.csv")
    
    # Download posters
    img_dir = "data/images"
    downloaded, failed = download_all_posters(df, img_dir)
    
    if downloaded == 0:
        print("❌ No posters downloaded. Please check the poster URLs in your dataset.")
        return
    
    print(f"✅ Downloaded {downloaded} posters to {img_dir}")
    
    # Filter dataset to only include movies with downloaded posters
    available_posters = [f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
    df_filtered = df[df['id'].astype(str).isin(available_posters)].copy()
    
    print(f"📊 Using {len(df_filtered)} movies with available posters")
    
    # Create datasets
    print("📦 Creating PyTorch datasets...")
    
    # Split into train/val
    train_size = int(0.8 * len(df_filtered))
    val_size = len(df_filtered) - train_size
    
    train_df = df_filtered.iloc[:train_size]
    val_df = df_filtered.iloc[train_size:]
    
    train_dataset = MoviePosterDataset(train_df, img_dir, transform=IMG_TF_TRAIN)
    val_dataset = MoviePosterDataset(val_df, img_dir, transform=IMG_TF)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"📦 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Initialize model
    print("🏗️  Initializing ResNet model...")
    model = create_resnet_model(num_classes=10, pretrained=True).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Train model
    print("🎯 Starting training...")
    train_losses, val_losses, train_f1s, val_f1s = train_model(
        model, train_loader, val_loader, epochs=25, device=device
    )
    
    # Plot results
    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s)
    
    # Save final model
    print("💾 Saving final model...")
    torch.save(model.state_dict(), 'models/poster_genre_classifier.pth')
    print("✅ Model saved successfully!")
    
    # Test prediction
    print("🧪 Testing model...")
    model.eval()
    
    # Test with a sample image
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_images, test_labels = next(iter(test_loader))
    
    with torch.no_grad():
        test_output = torch.sigmoid(model(test_images.to(device)))[0].cpu().numpy()
    
    genre_columns = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                     'Horror', 'Documentary', 'Animation', 'Music', 'Crime']
    
    top3 = test_output.argsort()[-3:][::-1]
    print("🎬 Test prediction:")
    for i, idx in enumerate(top3):
        print(f"   {i+1}. {genre_columns[idx]}: {test_output[idx]:.3f}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    
    main()

