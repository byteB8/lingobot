"""
Output formatting module for healthcare chatbot.
Handles response formatting, source attribution, and final output preparation.
"""

from typing import Dict, List, Any, Optional
import re


class OutputFormatter:
    """Handles output formatting and post-processing."""
    
    def __init__(self):
        """Initialize the output formatter."""
        self.max_response_length = 500
        self.min_response_length = 20
    
    def format_response(self, raw_response: str) -> str:
        """
        Format the raw response from the LLM.
        
        Args:
            raw_response: Raw response from the LLM
            
        Returns:
            Formatted response
        """
        if not raw_response:
            return "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
        
        # Clean up the response
        response = raw_response.strip()
        
        # Remove any incomplete sentences at the end
        response = re.sub(r'[.!?]\s*$', '.', response)
        
        # Ensure proper capitalization
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        
        # Truncate if too long
        if len(response) > self.max_response_length:
            # Find the last complete sentence
            sentences = re.split(r'[.!?]', response)
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]) + '.'
            else:
                response = response[:self.max_response_length] + '...'
        
        # Ensure minimum length
        if len(response) < self.min_response_length:
            response = "I'm sorry, I don't have enough information to provide a complete answer. Please consult a healthcare professional for more detailed guidance."
        
        return response
    
    def add_sources(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """
        Add source attribution to the response.
        
        Args:
            response: Formatted response
            sources: List of source information
            
        Returns:
            Response with source attribution
        """
        if not sources:
            return response
        
        # Extract unique sources
        unique_sources = set()
        for source in sources:
            if 'source' in source and source['source']:
                unique_sources.add(source['source'])
        
        if unique_sources:
            source_text = f"\n\nSources: {', '.join(sorted(unique_sources))}"
            response += source_text
        
        return response
    
    def ensure_disclaimer(self, response: str) -> str:
        """
        Ensure the response has the required disclaimer.
        
        Args:
            response: Response text
            
        Returns:
            Response with disclaimer
        """
        disclaimer = "This is for educational purposes only and not a substitute for professional medical advice."
        
        # Check if disclaimer already exists
        if disclaimer.lower() not in response.lower():
            response += f"\n\nDisclaimer: {disclaimer}"
        
        return response
    
    def format_error_response(self, error_type: str, query: str) -> str:
        """
        Format error responses.
        
        Args:
            error_type: Type of error
            query: Original query
            
        Returns:
            Formatted error response
        """
        error_responses = {
            'no_context': "I'm sorry, I don't have specific information about that. Please consult a healthcare professional for personalized advice.",
            'low_confidence': "I'm not entirely sure about that topic. Please consult a healthcare professional for accurate information.",
            'safety_filter': "I cannot provide specific medical advice. Please consult a healthcare professional for personalized guidance.",
            'generation_error': "I'm having trouble generating a response. Please try rephrasing your question or consult a healthcare professional."
        }
        
        base_response = error_responses.get(error_type, error_responses['generation_error'])
        
        # Add disclaimer
        return self.ensure_disclaimer(base_response)
    
    def format_greeting_response(self, query: str) -> str:
        """
        Format greeting responses.
        
        Args:
            query: User's greeting query
            
        Returns:
            Formatted greeting response
        """
        greeting_responses = [
            "Hello! I'm LingoBot, your healthcare information assistant. How can I help you today?",
            "Hi there! I'm here to help with general health questions, lifestyle tips, and hospital information. What would you like to know?",
            "Good day! I'm LingoBot. I can help you with health FAQs, wellness tips, or hospital information. What can I assist you with?",
            "Hello! Welcome to LingoBot. I'm here to provide general health information and lifestyle guidance. How can I help you?"
        ]
        
        import random
        response = random.choice(greeting_responses)
        
        return self.ensure_disclaimer(response)
    
    def format_final_output(self, response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the final output with metadata.
        
        Args:
            response: Formatted response
            metadata: Response metadata
            
        Returns:
            Complete output dictionary
        """
        return {
            'response': response,
            'intent': metadata.get('intent', 'Unknown'),
            'confidence': metadata.get('confidence', 0.0),
            'handler': metadata.get('handler', 'unknown'),
            'sources': metadata.get('sources', []),
            'timestamp': metadata.get('timestamp', ''),
            'query': metadata.get('query', ''),
            'context_count': metadata.get('context_count', 0)
        }
