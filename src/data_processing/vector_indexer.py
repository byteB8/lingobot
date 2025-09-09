"""
Vector indexing module for healthcare chatbot.
Creates FAISS indices for FAQ and lifestyle tips data.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle


class VectorIndexer:
    """Handles vector indexing of text data using sentence transformers and FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector indexer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.faq_index = None
        self.tips_index = None
        self.faq_data = []
        self.tips_data = []
        
    def load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def create_faq_index(self, faq_data: List[Dict[str, str]]) -> None:
        """
        Create FAISS index for FAQ data.
        
        Args:
            faq_data: List of FAQ dictionaries with 'question' and 'answer' keys
        """
        print("Creating FAQ index...")
        
        # Prepare texts for embedding (combine question and answer)
        texts = []
        for faq in faq_data:
            combined_text = f"{faq['question']} {faq['answer']}"
            texts.append(combined_text)
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faq_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.faq_index.add(embeddings.astype('float32'))
        
        # Store original data
        self.faq_data = faq_data
        
        print(f"FAQ index created with {len(faq_data)} entries")
    
    def create_tips_index(self, tips_data: List[Dict[str, str]]) -> None:
        """
        Create FAISS index for lifestyle tips data.
        
        Args:
            tips_data: List of tip dictionaries with 'tip' key
        """
        print("Creating lifestyle tips index...")
        
        # Prepare texts for embedding
        texts = [tip['tip'] for tip in tips_data]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.tips_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.tips_index.add(embeddings.astype('float32'))
        
        # Store original data
        self.tips_data = tips_data
        
        print(f"Lifestyle tips index created with {len(tips_data)} entries")
    
    def search_faq(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search FAQ index for relevant entries.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant FAQ entries with scores
        """
        if self.faq_index is None:
            raise ValueError("FAQ index not created. Call create_faq_index first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.faq_index.search(query_embedding.astype('float32'), top_k)
        
        # Return results with original data
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.faq_data):
                result = self.faq_data[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search_tips(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search lifestyle tips index for relevant entries.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant tip entries with scores
        """
        if self.tips_index is None:
            raise ValueError("Tips index not created. Call create_tips_index first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.tips_index.search(query_embedding.astype('float32'), top_k)
        
        # Return results with original data
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.tips_data):
                result = self.tips_data[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save_indices(self, output_dir: str) -> None:
        """
        Save FAISS indices and metadata to disk.
        
        Args:
            output_dir: Directory to save indices
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save FAQ index
        if self.faq_index is not None:
            faiss.write_index(self.faq_index, str(output_path / "faq_index.faiss"))
            with open(output_path / "faq_data.pkl", 'wb') as f:
                pickle.dump(self.faq_data, f)
        
        # Save tips index
        if self.tips_index is not None:
            faiss.write_index(self.tips_index, str(output_path / "tips_index.faiss"))
            with open(output_path / "tips_data.pkl", 'wb') as f:
                pickle.dump(self.tips_data, f)
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'faq_count': len(self.faq_data) if self.faq_data else 0,
            'tips_count': len(self.tips_data) if self.tips_data else 0
        }
        
        with open(output_path / "index_metadata.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Indices saved to {output_dir}")
    
    def load_indices(self, input_dir: str) -> None:
        """
        Load FAISS indices and metadata from disk.
        
        Args:
            input_dir: Directory containing saved indices
        """
        input_path = Path(input_dir)
        
        # Load model
        self.load_model()
        
        # Load metadata
        with open(input_path / "index_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load FAQ index
        if (input_path / "faq_index.faiss").exists():
            self.faq_index = faiss.read_index(str(input_path / "faq_index.faiss"))
            with open(input_path / "faq_data.pkl", 'rb') as f:
                self.faq_data = pickle.load(f)
            print(f"Loaded FAQ index with {len(self.faq_data)} entries")
        
        # Load tips index
        if (input_path / "tips_index.faiss").exists():
            self.tips_index = faiss.read_index(str(input_path / "tips_index.faiss"))
            with open(input_path / "tips_data.pkl", 'rb') as f:
                self.tips_data = pickle.load(f)
            print(f"Loaded tips index with {len(self.tips_data)} entries")
        
        print("Indices loaded successfully!")


if __name__ == "__main__":
    # Test the vector indexer
    from data_preprocessor import DataPreprocessor
    
    # Load processed data
    with open("data/processed_data.json", 'r') as f:
        processed_data = json.load(f)
    
    # Create indexer
    indexer = VectorIndexer()
    
    # Create indices
    indexer.create_faq_index(processed_data['faq'])
    indexer.create_tips_index(processed_data['lifestyle_tips'])
    
    # Save indices
    indexer.save_indices("data/indices")
    
    # Test search
    print("\nTesting FAQ search...")
    faq_results = indexer.search_faq("What is glaucoma?", top_k=2)
    for result in faq_results:
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Q: {result['question'][:100]}...")
        print(f"A: {result['answer'][:100]}...")
        print()
    
    print("Testing tips search...")
    tips_results = indexer.search_tips("healthy diet", top_k=2)
    for result in tips_results:
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Tip: {result['tip'][:100]}...")
        print()

