"""
Query routing module for healthcare chatbot.
Routes queries to appropriate handlers based on intent.
"""

from typing import Dict, List, Any, Optional
from .intent_classifier import IntentClassifier
from ..data_processing.vector_indexer import VectorIndexer


class QueryRouter:
    """Routes user queries to appropriate handlers based on intent."""
    
    def __init__(self, vector_indexer: VectorIndexer, hospital_data: Dict[str, Any]):
        """
        Initialize the query router.
        
        Args:
            vector_indexer: Vector indexer for FAQ and tips search
            hospital_data: Hospital information data
        """
        self.intent_classifier = IntentClassifier()
        self.vector_indexer = vector_indexer
        self.hospital_data = hospital_data
        
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route a user query to the appropriate handler.
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary containing routing information and context
        """
        # Classify intent
        intent, confidence = self.intent_classifier.classify_with_rules(query)
        
        # Route based on intent
        if intent == "General Health FAQ":
            return self._handle_faq_query(query, intent, confidence)
        elif intent == "Healthy Lifestyle Tip":
            return self._handle_tips_query(query, intent, confidence)
        elif intent == "Hospital Information":
            return self._handle_hospital_query(query, intent, confidence)
        elif intent == "Greeting or Chit-chat":
            return self._handle_greeting_query(query, intent, confidence)
        else:
            return self._handle_fallback_query(query, intent, confidence)
    
    def _handle_faq_query(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Handle general health FAQ queries."""
        try:
            # Search FAQ index
            faq_results = self.vector_indexer.search_faq(query, top_k=3)
            
            return {
                'intent': intent,
                'confidence': confidence,
                'handler': 'faq',
                'context': faq_results,
                'query': query
            }
        except Exception as e:
            return {
                'intent': intent,
                'confidence': confidence,
                'handler': 'faq',
                'context': [],
                'query': query,
                'error': str(e)
            }
    
    def _handle_tips_query(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Handle healthy lifestyle tips queries."""
        try:
            # Search tips index
            tips_results = self.vector_indexer.search_tips(query, top_k=3)
            
            return {
                'intent': intent,
                'confidence': confidence,
                'handler': 'tips',
                'context': tips_results,
                'query': query
            }
        except Exception as e:
            return {
                'intent': intent,
                'confidence': confidence,
                'handler': 'tips',
                'context': [],
                'query': query,
                'error': str(e)
            }
    
    def _handle_hospital_query(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Handle hospital information queries."""
        try:
            # Extract relevant hospital information
            hospital_context = self._extract_hospital_info(query)
            
            return {
                'intent': intent,
                'confidence': confidence,
                'handler': 'hospital',
                'context': hospital_context,
                'query': query
            }
        except Exception as e:
            return {
                'intent': intent,
                'confidence': confidence,
                'handler': 'hospital',
                'context': [],
                'query': query,
                'error': str(e)
            }
    
    def _handle_greeting_query(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Handle greeting and chit-chat queries."""
        return {
            'intent': intent,
            'confidence': confidence,
            'handler': 'greeting',
            'context': [],
            'query': query
        }
    
    def _handle_fallback_query(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """Handle fallback queries with low confidence."""
        return {
            'intent': intent,
            'confidence': confidence,
            'handler': 'fallback',
            'context': [],
            'query': query
        }
    
    def _extract_hospital_info(self, query: str) -> List[Dict[str, Any]]:
        """Extract relevant hospital information based on query keywords."""
        query_lower = query.lower()
        hospital_context = []
        
        # Check for specific information types
        if any(word in query_lower for word in ['hours', 'time', 'open', 'closed', 'working']):
            hospital_context.append({
                'type': 'working_hours',
                'data': self.hospital_data['working_hours']
            })
        
        if any(word in query_lower for word in ['contact', 'phone', 'call', 'number']):
            hospital_context.append({
                'type': 'contact',
                'data': self.hospital_data['basic_info']['contact']
            })
        
        if any(word in query_lower for word in ['address', 'location', 'where']):
            hospital_context.append({
                'type': 'address',
                'data': self.hospital_data['basic_info']['address']
            })
        
        if any(word in query_lower for word in ['appointment', 'book', 'booking']):
            hospital_context.append({
                'type': 'appointment',
                'data': self.hospital_data['appointment']
            })
        
        if any(word in query_lower for word in ['department', 'specialty', 'specialist']):
            hospital_context.append({
                'type': 'departments',
                'data': self.hospital_data['departments']
            })
        
        # If no specific info found, return general hospital info
        if not hospital_context:
            hospital_context.append({
                'type': 'general',
                'data': self.hospital_data['basic_info']
            })
        
        return hospital_context
