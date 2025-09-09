#!/usr/bin/env python3
"""
Script to create vector indices from processed data.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing.vector_indexer import VectorIndexer

def main():
    """Create vector indices from processed data."""
    
    # Load processed data
    with open("data/processed_data.json", 'r') as f:
        processed_data = json.load(f)
    
    # Create indexer
    indexer = VectorIndexer()
    
    # Create indices
    print("Creating FAQ index...")
    indexer.create_faq_index(processed_data['faq'])
    
    print("Creating tips index...")
    indexer.create_tips_index(processed_data['lifestyle_tips'])
    
    # Save indices
    indexer.save_indices("data/indices")
    print("Indices created successfully!")

if __name__ == "__main__":
    main()
