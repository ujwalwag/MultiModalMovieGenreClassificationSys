import os
import requests
from tqdm import tqdm
import zipfile

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    """Download GloVe embeddings"""
    print("🚀 Downloading GloVe embeddings...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # GloVe 6B 100d embeddings
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_zip = "data/glove.6B.zip"
    glove_file = "data/glove.6B.100d.txt"
    
    # Check if already downloaded
    if os.path.exists(glove_file):
        print("✅ GloVe embeddings already exist!")
        return
    
    # Download GloVe embeddings
    print("📥 Downloading GloVe 6B embeddings (400MB)...")
    download_file(glove_url, glove_zip)
    
    # Extract the 100d embeddings
    print("📦 Extracting 100d embeddings...")
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extract("glove.6B.100d.txt", "data/")
    
    # Clean up zip file
    os.remove(glove_zip)
    
    print("✅ GloVe embeddings downloaded and extracted successfully!")
    print(f"📁 File location: {glove_file}")

if __name__ == "__main__":
    main()

