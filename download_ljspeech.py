"""Script to download LJSpeech dataset."""

from src.iris.datasets import download_ljspeech

if __name__ == "__main__":
    print("Downloading LJSpeech dataset...")
    ljspeech_path = download_ljspeech(data_dir="./data", extract=True, remove_tar=False)
    print(f"\nDataset downloaded and extracted to: {ljspeech_path}")

