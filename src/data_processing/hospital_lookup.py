"""
Hospital data lookup module for healthcare chatbot.
Provides direct lookup functionality for hospital information.
"""

from typing import Dict, List, Any, Optional
import re
from collections import defaultdict


class HospitalLookup:
    """Handles direct lookup of hospital information."""
    
    def __init__(self, hospital_data: Dict[str, Any]):
        """
        Initialize hospital lookup with structured data.
        
        Args:
            hospital_data: Preprocessed hospital data
        """
        self.hospital_data = hospital_data
        self._build_lookup_indexes()
    
    def _build_lookup_indexes(self):
        """Build search indexes for faster lookup."""
        # Build keyword indexes for different information types
        self.working_hours_keywords = [
            'hours', 'time', 'open', 'closed', 'working', 'schedule',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'weekend', 'weekday'
        ]
        
        self.contact_keywords = [
            'contact', 'phone', 'call', 'number', 'email', 'reach',
            'telephone', 'mobile', 'contact us'
        ]
        
        self.address_keywords = [
            'address', 'location', 'where', 'find', 'directions',
            'place', 'building', 'street', 'road'
        ]
        
        self.appointment_keywords = [
            'appointment', 'book', 'booking', 'schedule', 'visit',
            'consultation', 'meeting', 'reserve'
        ]
        
        self.department_keywords = [
            'department', 'specialty', 'specialist', 'service',
            'clinic', 'unit', 'center', 'division'
        ]
        
        self.emergency_keywords = [
            'emergency', 'urgent', 'crisis', 'immediate', 'help',
            'ambulance', '911', 'critical'
        ]
        
        # Build FAQ search index
        self.faq_index = {}
        for faq in self.hospital_data.get('faq', []):
            question = faq.get('question', '').lower()
            # Index by keywords in question
            for word in question.split():
                if word not in self.faq_index:
                    self.faq_index[word] = []
                self.faq_index[word].append(faq)
    
    def search_working_hours(self, query: str) -> Dict[str, Any]:
        """Search for working hours information."""
        query_lower = query.lower()
        
        # Check if query is about working hours
        if any(keyword in query_lower for keyword in self.working_hours_keywords):
            return {
                'type': 'working_hours',
                'data': self.hospital_data['working_hours'],
                'confidence': 0.9
            }
        
        return None
    
    def search_contact_info(self, query: str) -> Dict[str, Any]:
        """Search for contact information."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.contact_keywords):
            return {
                'type': 'contact',
                'data': self.hospital_data['basic_info']['contact'],
                'confidence': 0.9
            }
        
        return None
    
    def search_address(self, query: str) -> Dict[str, Any]:
        """Search for address information."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.address_keywords):
            return {
                'type': 'address',
                'data': self.hospital_data['basic_info']['address'],
                'confidence': 0.9
            }
        
        return None
    
    def search_appointment_info(self, query: str) -> Dict[str, Any]:
        """Search for appointment booking information."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.appointment_keywords):
            return {
                'type': 'appointment',
                'data': self.hospital_data['appointment'],
                'confidence': 0.9
            }
        
        return None
    
    def search_departments(self, query: str) -> Dict[str, Any]:
        """Search for department information."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.department_keywords):
            return {
                'type': 'departments',
                'data': self.hospital_data['departments'],
                'confidence': 0.9
            }
        
        return None
    
    def search_emergency_info(self, query: str) -> Dict[str, Any]:
        """Search for emergency services information."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in self.emergency_keywords):
            return {
                'type': 'emergency',
                'data': self.hospital_data['appointment']['emergency'],
                'confidence': 0.95
            }
        
        return None
    
    def search_faq(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search hospital FAQ using keyword matching."""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Score FAQ entries based on keyword matches
        faq_scores = defaultdict(float)
        
        for faq in self.hospital_data.get('faq', []):
            question = faq.get('question', '').lower()
            answer = faq.get('answer', '').lower()
            
            # Calculate score based on word matches
            score = 0
            for word in query_words:
                if word in question:
                    score += 2  # Higher weight for question matches
                if word in answer:
                    score += 1  # Lower weight for answer matches
            
            if score > 0:
                faq_scores[tuple(faq.items())] = score
        
        # Sort by score and return top results
        sorted_faqs = sorted(faq_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for faq_items, score in sorted_faqs[:top_k]:
            faq_dict = dict(faq_items)
            faq_dict['similarity_score'] = score / len(query_words)  # Normalize score
            results.append(faq_dict)
        
        return results
    
    def search_general_info(self, query: str) -> Dict[str, Any]:
        """Search for general hospital information."""
        return {
            'type': 'general',
            'data': self.hospital_data['basic_info'],
            'confidence': 0.7
        }
    
    def lookup(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform comprehensive hospital information lookup.
        
        Args:
            query: User's query
            
        Returns:
            List of relevant hospital information
        """
        results = []
        
        # Try different search methods in order of specificity
        search_methods = [
            self.search_emergency_info,
            self.search_working_hours,
            self.search_contact_info,
            self.search_address,
            self.search_appointment_info,
            self.search_departments
        ]
        
        for method in search_methods:
            result = method(query)
            if result:
                results.append(result)
        
        # If no specific info found, try FAQ search
        if not results:
            faq_results = self.search_faq(query)
            if faq_results:
                results.extend([{
                    'type': 'faq',
                    'data': faq,
                    'confidence': faq.get('similarity_score', 0.5)
                } for faq in faq_results])
        
        # If still no results, return general info
        if not results:
            results.append(self.search_general_info(query))
        
        return results
    
    def get_all_departments(self) -> List[str]:
        """Get list of all available departments."""
        return self.hospital_data.get('departments', [])
    
    def get_working_hours(self) -> Dict[str, str]:
        """Get working hours information."""
        return self.hospital_data.get('working_hours', {})
    
    def get_contact_info(self) -> Dict[str, str]:
        """Get contact information."""
        return self.hospital_data['basic_info'].get('contact', {})
    
    def get_address(self) -> str:
        """Get hospital address."""
        return self.hospital_data['basic_info'].get('address', '')
    
    def get_appointment_info(self) -> Dict[str, Any]:
        """Get appointment booking information."""
        return self.hospital_data.get('appointment', {})
