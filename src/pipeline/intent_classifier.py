"""
Intent classification module for healthcare chatbot.
Uses both TF-IDF and zero-shot classification to determine user intent.
"""

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import re
import numpy as np


class TFIDFIntentClassifier:
    """Handles intent classification using TF-IDF and cosine similarity."""
    
    def __init__(self):
        """Initialize the TF-IDF intent classifier."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Golden sentences for each intent
        self.golden_sentences = {
            "General Health FAQ": [
                "What are the symptoms of flu?",
                "How do I treat a headache?",
                "What causes high blood pressure?",
                "Tell me about diabetes symptoms",
                "How to prevent heart disease?",
                "What are the signs of infection?",
                "How to manage stress?",
                "What is the treatment for fever?"
            ],
            "Healthy Lifestyle Tip": [
                "Give me advice on healthy eating",
                "How to exercise properly?",
                "Tips for better sleep",
                "Healthy diet recommendations",
                "Exercise routine suggestions",
                "Sleep hygiene tips",
                "Nutritional advice",
                "Fitness recommendations"
            ],
            "Hospital Information": [
                "Tell me about hospital hours",
                "What is the contact information?",
                "How to book an appointment?",
                "What departments are available?",
                "Hospital address and location",
                "Emergency services information",
                "Working hours and schedule",
                "Contact phone number"
            ],
            "Greeting or Chit-chat": [
                "Hello, how are you?",
                "Hi there, good morning",
                "Thanks for your help",
                "Goodbye and take care",
                "How is everything?",
                "Nice to meet you",
                "Have a great day",
                "See you later"
            ]
        }
        
        self.intent_vectors = None
        self.intent_labels = None
        self._fit_vectorizer()
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on golden sentences."""
        all_sentences = []
        all_labels = []
        
        for intent, sentences in self.golden_sentences.items():
            all_sentences.extend(sentences)
            all_labels.extend([intent] * len(sentences))
        
        # Fit vectorizer and transform sentences
        self.intent_vectors = self.vectorizer.fit_transform(all_sentences)
        self.intent_labels = all_labels
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize the user query."""
        if not isinstance(query, str):
            return ""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase
        query = query.lower()
        
        return query
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify the user's query into one of the predefined intents.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        cleaned_query = self.clean_query(query)
        
        if not cleaned_query:
            return "Fallback", 0.0
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Calculate cosine similarity with all golden sentences
        similarities = cosine_similarity(query_vector, self.intent_vectors).flatten()
        
        # Get the highest similarity score
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        predicted_intent = self.intent_labels[max_similarity_idx]
        
        # Apply confidence threshold
        confidence_threshold = 0.3
        if max_similarity >= confidence_threshold:
            return predicted_intent, float(max_similarity)
        else:
            return "Fallback", float(max_similarity)
    
    def get_intent_confidence(self, query: str) -> Dict[str, float]:
        """
        Get confidence scores for all intents.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary mapping intents to confidence scores
        """
        cleaned_query = self.clean_query(query)
        
        if not cleaned_query:
            return {intent: 0.0 for intent in self.golden_sentences.keys()}
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([cleaned_query])
        
        # Calculate cosine similarity with all golden sentences
        similarities = cosine_similarity(query_vector, self.intent_vectors).flatten()
        
        # Group similarities by intent
        intent_scores = {}
        for intent in self.golden_sentences.keys():
            intent_scores[intent] = 0.0
        
        for i, similarity in enumerate(similarities):
            intent = self.intent_labels[i]
            intent_scores[intent] = max(intent_scores[intent], similarity)
        
        return intent_scores


class ZSIntentClassifier:
    """Handles intent classification using zero-shot classification."""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the intent classifier.
        
        Args:
            model_name: Name of the zero-shot classification model
        """
        self.model_name = model_name
        self.classifier = None
        self.candidate_labels = [
            "General Health FAQ",
            "Healthy Lifestyle Tip", 
            "Hospital Information",
            "Greeting or Chit-chat"
        ]
        self.hypothesis_template = "This example is about {}."
        self.confidence_threshold = 0.70
        
    def load_model(self):
        """Load the zero-shot classification model."""
        print(f"Loading intent classification model: {self.model_name}")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name
        )
        print("Intent classification model loaded!")
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize the user query."""
        if not isinstance(query, str):
            return ""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for better classification
        query = query.lower()
        
        return query
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify the user's query into one of the predefined intents.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        if self.classifier is None:
            self.load_model()
        
        # Clean the query
        cleaned_query = self.clean_query(query)
        
        if not cleaned_query:
            return "Fallback", 0.0
        
        # Classify the query
        result = self.classifier(
            cleaned_query,
            self.candidate_labels,
            hypothesis_template=self.hypothesis_template
        )
        
        # Get the top result
        top_intent = result['labels'][0]
        confidence_score = result['scores'][0]
        
        # Apply confidence threshold
        if confidence_score >= self.confidence_threshold:
            return top_intent, confidence_score
        else:
            return "Fallback", confidence_score
    
    def get_intent_confidence(self, query: str) -> Dict[str, float]:
        """
        Get confidence scores for all intents.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary mapping intents to confidence scores
        """
        if self.classifier is None:
            self.load_model()
        
        cleaned_query = self.clean_query(query)
        
        if not cleaned_query:
            return {label: 0.0 for label in self.candidate_labels}
        
        result = self.classifier(
            cleaned_query,
            self.candidate_labels,
            hypothesis_template=self.hypothesis_template
        )
        
        return dict(zip(result['labels'], result['scores']))
    
    def is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting or chit-chat."""
        greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
            r'\b(how are you|how do you do|what\'s up)\b',
            r'\b(thanks|thank you|bye|goodbye)\b',
            r'\b(yes|no|ok|okay|sure)\b'
        ]
        
        cleaned_query = self.clean_query(query)
        
        for pattern in greeting_patterns:
            if re.search(pattern, cleaned_query, re.IGNORECASE):
                return True
        
        return False
    
    def is_hospital_query(self, query: str) -> bool:
        """Check if the query is about hospital information."""
        hospital_keywords = [
            'hospital', 'clinic', 'doctor', 'appointment', 'booking',
            'hours', 'contact', 'phone', 'email', 'address', 'location',
            'department', 'emergency', 'services', 'working', 'open',
            'closed', 'saturday', 'sunday', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday'
        ]
        
        cleaned_query = self.clean_query(query)
        
        for keyword in hospital_keywords:
            if keyword in cleaned_query:
                return True
        
        return False
    
    def classify_with_rules(self, query: str) -> Tuple[str, float]:
        """
        Classify intent using both ML model and rule-based approach.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        # First check rule-based patterns
        if self.is_greeting(query):
            return "Greeting or Chit-chat", 0.95
        
        if self.is_hospital_query(query):
            return "Hospital Information", 0.90
        
        # Fall back to ML classification
        return self.classify_intent(query)
