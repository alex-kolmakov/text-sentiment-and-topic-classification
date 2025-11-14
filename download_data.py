#!/usr/bin/env python3
"""
Data Download Script for Ubuntu Dialogue Corpus

This script downloads the Ubuntu Dialogue Corpus dataset from Kaggle
using kagglehub and prints the path to the downloaded files.
"""

import kagglehub
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_ubuntu_dialogue_corpus():
    """
    Download the Ubuntu Dialogue Corpus from Kaggle.
    
    Returns:
        Path: Path to the downloaded dataset files
    """
    logger.info("Starting download of Ubuntu Dialogue Corpus from Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("rtatman/ubuntu-dialogue-corpus")
        
        logger.info(f"✓ Dataset downloaded successfully!")
        logger.info(f"Path to dataset files: {path}")
        
        # List files in the downloaded directory
        dataset_path = Path(path)
        if dataset_path.exists():
            files = list(dataset_path.glob("*"))
            logger.info(f"\nFound {len(files)} file(s) in dataset:")
            for file in files:
                logger.info(f"  - {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)")
        
        return path
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Ubuntu Dialogue Corpus - Data Download")
    print("=" * 60)
    print()
    
    try:
        dataset_path = download_ubuntu_dialogue_corpus()
        print()
        print("=" * 60)
        print(f"SUCCESS! Dataset available at: {dataset_path}")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"FAILED: {e}")
        print("=" * 60)
        exit(1)
