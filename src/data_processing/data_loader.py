"""
Data loader module for healthcare chatbot.
Loads preprocessed and indexed data at application startup.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from .vector_indexer import VectorIndexer


class DataLoader:
    """Handles loading of preprocessed and indexed data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_data = None
        self.vector_indexer = None
        
    def load_processed_data(self) -> Dict[str, Any]:
        """Load preprocessed data from JSON file."""
        processed_file = self.data_dir / "processed_data.json"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data not found at {processed_file}")
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)
        
        print(f"Loaded processed data: {self.processed_data['metadata']}")
        return self.processed_data
    
    def load_vector_indices(self) -> VectorIndexer:
        """Load vector indices from disk."""
        indices_dir = self.data_dir / "indices"
        
        if not indices_dir.exists():
            raise FileNotFoundError(f"Vector indices not found at {indices_dir}")
        
        self.vector_indexer = VectorIndexer()
        self.vector_indexer.load_indices(str(indices_dir))
        
        return self.vector_indexer
    
    def load_all_data(self) -> Dict[str, Any]:
        """Load all data (processed + vector indices)."""
        processed_data = self.load_processed_data()
        vector_indexer = self.load_vector_indices()
        
        return {
            'processed_data': processed_data,
            'vector_indexer': vector_indexer
        }
