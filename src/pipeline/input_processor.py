"""
Input processing module for healthcare chatbot.
Handles query cleaning, validation, and preprocessing.
"""

import re
from typing import Dict, List, Optional, Tuple


class InputProcessor:
    """Handles input processing and validation."""
    
    def __init__(self, max_length: int = 500):
        """
        Initialize the input processor.
        
        Args:
            max_length: Maximum allowed query length
        """
        self.max_length = max_length
        self.min_length = 2
        
    def clean_query(self, user_input: str) -> str:
        """
        Clean and normalize user input.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Cleaned query string
        """
        if not isinstance(user_input, str):
            return ""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', user_input.strip())
        
        # Remove special characters but keep basic punctuation
        query = re.sub(r'[^\w\s.,!?;:-]', '', query)
        
        # Truncate if too long
        if len(query) > self.max_length:
            query = query[:self.max_length].rsplit(' ', 1)[0]
        
        return query
    
    def validate_input(self, query: str) -> Tuple[bool, str]:
        """
        Validate user input.
        
        Args:
            query: User query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query cannot be empty"
        
        if len(query) < self.min_length:
            return False, f"Query too short (minimum {self.min_length} characters)"
        
        if len(query) > self.max_length:
            return False, f"Query too long (maximum {self.max_length} characters)"
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(emergency|urgent|help|dying|death|suicide)\b',
            r'\b(prescription|medication|drug|pill)\b',
            r'\b(diagnosis|diagnose|symptom|pain|hurt)\b'
        ]
        
        query_lower = query.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, query_lower):
                return True, "Query contains medical terms - please consult a healthcare professional for medical advice"
        
        return True, ""
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query for better routing.
        
        Args:
            query: User query
            
        Returns:
            List of extracted keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def preprocess_query(self, user_input: str) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for user input.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Dictionary containing processed query and metadata
        """
        # Clean the query
        cleaned_query = self.clean_query(user_input)
        
        # Validate the query
        is_valid, error_message = self.validate_input(cleaned_query)
        
        # Extract keywords
        keywords = self.extract_keywords(cleaned_query)
        
        return {
            'original_input': user_input,
            'cleaned_query': cleaned_query,
            'is_valid': is_valid,
            'error_message': error_message,
            'keywords': keywords,
            'length': len(cleaned_query)
        }
