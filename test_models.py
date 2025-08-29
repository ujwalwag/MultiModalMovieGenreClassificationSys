#!/usr/bin/env python3
"""
Test script for MMGCS models
Tests both text and image models to ensure they work correctly
"""

import os
import torch
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

# Import the models from app.py
from app import GenreLSTM, GENRE_COLUMNS, IMG_TF

def test_text_model():
    """Test the text classification model"""
    print("🧪 Testing Text Model...")
    
    # Check if model files exist
    if not os.path.exists('models/genre_classifier.pth'):
        print("❌ Text model not found!")
        return False
    
    if not os.path.exists('models/tokenizer.pickle'):
        print("❌ Tokenizer not found!")
        return False
    
    if not os.path.exists('models/embedding_matrix.npy'):
        print("❌ Embedding matrix not found!")
        return False
    
    try:
        # Load model components
        with open('models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        embedding_matrix = np.load('models/embedding_matrix.npy')
        model_state = torch.load('models/genre_classifier.pth', map_location='cpu')
        
        # Initialize model
        model = GenreLSTM(embedding_matrix)
        model.load_state_dict(model_state)
        model.eval()
        
        # Test prediction
        test_texts = [
            "A young boy discovers he has magical powers and goes on an adventure.",
            "A romantic comedy about two people who fall in love despite their differences.",
            "A thrilling action movie with car chases and explosions.",
            "A documentary about wildlife in Africa."
        ]
        
        for text in test_texts:
            # Tokenize
            seq = tokenizer.texts_to_sequences([text])[0][:200]
            arr = np.zeros((1, 200), dtype=np.int64)
            arr[0, :len(seq)] = seq
            tensor = torch.tensor(arr, dtype=torch.long)
            
            # Predict
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor))[0].numpy()
            
            top3 = probs.argsort()[-3:][::-1]
            print(f"\n🎬 '{text}'")
            for i, idx in enumerate(top3):
                print(f"   {i+1}. {GENRE_COLUMNS[idx]}: {probs[idx]:.3f}")
        
        print("✅ Text model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text model test failed: {str(e)}")
        return False

def test_image_model():
    """Test the image classification model"""
    print("\n🧪 Testing Image Model...")
    
    # Check if model file exists
    if not os.path.exists('models/poster_genre_classifier.pth'):
        print("❌ Image model not found!")
        return False
    
    try:
        # Load model
        from torchvision import models
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(GENRE_COLUMNS))
        state = torch.load('models/poster_genre_classifier.pth', map_location='cpu')
        model.load_state_dict(state, strict=False)
        model.eval()
        
        # Check if we have any test images
        img_dir = "data/images"
        if not os.path.exists(img_dir):
            print("❌ No images directory found!")
            return False
        
        image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        if not image_files:
            print("❌ No images found in data/images/")
            return False
        
        # Test with first available image
        test_img_path = os.path.join(img_dir, image_files[0])
        print(f"🖼️  Testing with: {image_files[0]}")
        
        # Load and transform image
        img = Image.open(test_img_path).convert("RGB")
        tensor = IMG_TF(img).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].numpy()
        
        top3 = probs.argsort()[-3:][::-1]
        print("🎬 Predictions:")
        for i, idx in enumerate(top3):
            print(f"   {i+1}. {GENRE_COLUMNS[idx]}: {probs[idx]:.3f}")
        
        print("✅ Image model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Image model test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🎬 MMGCS Model Testing")
    print("="*50)
    
    # Test text model
    text_success = test_text_model()
    
    # Test image model
    image_success = test_image_model()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print(f"   Text Model: {'✅ PASS' if text_success else '❌ FAIL'}")
    print(f"   Image Model: {'✅ PASS' if image_success else '❌ FAIL'}")
    
    if text_success and image_success:
        print("\n🎉 All tests passed! Your MMGCS system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the model files and try again.")
    
    print("="*50)

if __name__ == "__main__":
    main()

