#!/usr/bin/env python3
"""
Main training script for MMGCS project
Trains both text and image models using PyTorch
"""

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        return False

def main():
    """Main training orchestration"""
    print("🎬 MMGCS - Multi-Modal Genre Classification System")
    print("🔥 PyTorch-based Training Pipeline")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("data/strictly_balanced_top10.csv"):
        print("❌ Dataset not found! Please ensure you're in the project root directory.")
        print("   Expected: data/strictly_balanced_top10.csv")
        return
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    
    # Step 1: Download GloVe embeddings
    print("\n📚 Step 1: Downloading GloVe embeddings...")
    if not os.path.exists("data/glove.6B.100d.txt"):
        success = run_script("scripts/download_glove.py", "GloVe embeddings download")
        if not success:
            print("⚠️  GloVe download failed. Continuing with random embeddings...")
    else:
        print("✅ GloVe embeddings already exist!")
    
    # Step 2: Train text model
    print("\n📝 Step 2: Training text model...")
    success_text = run_script("scripts/text/train_text_model.py", "Text model training")
    
    if not success_text:
        print("❌ Text model training failed. Stopping pipeline.")
        return
    
    # Step 3: Train image model
    print("\n🖼️  Step 3: Training image model...")
    success_image = run_script("scripts/image/train_image_model.py", "Image model training")
    
    if not success_image:
        print("❌ Image model training failed.")
        print("⚠️  Text model trained successfully, but image model failed.")
        return
    
    # Step 4: Verify models
    print("\n🔍 Step 4: Verifying trained models...")
    models_to_check = [
        "models/genre_classifier.pth",
        "models/poster_genre_classifier.pth",
        "models/embedding_matrix.npy",
        "models/tokenizer.pickle"
    ]
    
    all_models_exist = True
    for model_path in models_to_check:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {model_path} - MISSING!")
            all_models_exist = False
    
    if all_models_exist:
        print("\n🎉 All models trained successfully!")
        print("🚀 Your MMGCS system is ready to use!")
        print("\n📁 Models saved in: models/")
        print("📊 Training plots saved in: plots/")
        print("🖼️  Movie posters downloaded to: data/images/")
    else:
        print("\n⚠️  Some models are missing. Please check the training logs.")
    
    print("\n" + "="*60)
    print("🏁 Training pipeline completed!")
    print("="*60)

if __name__ == "__main__":
    main()

