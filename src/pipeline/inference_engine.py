"""
Inference engine module for healthcare chatbot.
Handles LLM generation with prompt templates and safety filters.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Any, Optional
import re


class InferenceEngine:
    """Handles LLM inference with prompt engineering and safety filters."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct", use_quantization: bool = True):
        """
        Initialize the inference engine.
        
        Args:
            model_name: Name of the LLM model to use
            use_quantization: Whether to use 4-bit quantization for memory optimization
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prompt templates
        self.promptv1 = """Context: {retrieved_context}
Question: {user_question}
Answer:"""
        
        self.promptv2 = """You are a helpful and friendly Healthcare Information Assistant. Your name is LingoBot. Answer the user's question based ONLY on the context provided below. If the context does not contain the answer, say "I'm sorry, I don't have that information. Please consult a medical professional."

Guidelines:
- Provide factual, evidence-based information
- Use simple, accessible language
- Be supportive but not diagnostic
- Always include the educational disclaimer
- If uncertain, recommend consulting healthcare professionals

Context: {relevant_knowledge}
User Question: {user_query}

Response format:
- Direct answer to the question
- Additional helpful information if relevant
- Disclaimer: "This is for educational purposes only and not a substitute for professional medical advice.""""
        
        # Currently using promptv2
        self.current_prompt = self.promptv2
        
        # Safety keywords to filter out
        self.unsafe_keywords = [
            'prescription', 'medication', 'drug', 'pill', 'dosage',
            'diagnosis', 'diagnose', 'treat', 'treatment', 'cure',
            'emergency', 'urgent', 'call 911', 'ambulance',
            'surgery', 'operation', 'procedure'
        ]
        
    def load_model(self):
        """Load the LLM model and tokenizer."""
        print(f"Loading LLM model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with optional quantization
        if self.use_quantization and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        
        print("LLM model loaded successfully!")
    
    def format_context(self, context_data: List[Dict[str, Any]], intent: str) -> str:
        """
        Format context data into a readable string.
        
        Args:
            context_data: List of context items from retrieval
            intent: The classified intent
            
        Returns:
            Formatted context string
        """
        if not context_data:
            return "No relevant information found."
        
        context_parts = []
        
        if intent == "General Health FAQ":
            for item in context_data:
                context_parts.append(f"Q: {item.get('question', '')}")
                context_parts.append(f"A: {item.get('answer', '')}")
                if 'source' in item:
                    context_parts.append(f"Source: {item['source']}")
                context_parts.append("")
        
        elif intent == "Healthy Lifestyle Tip":
            for item in context_data:
                context_parts.append(f"Tip: {item.get('tip', '')}")
                if 'category' in item:
                    context_parts.append(f"Category: {item['category']}")
                if 'source' in item:
                    context_parts.append(f"Source: {item['source']}")
                context_parts.append("")
        
        elif intent == "Hospital Information":
            for item in context_data:
                if item.get('type') == 'working_hours':
                    context_parts.append("Working Hours:")
                    hours = item.get('data', {})
                    for day, time in hours.items():
                        context_parts.append(f"  {day.replace('_', ' ').title()}: {time}")
                
                elif item.get('type') == 'contact':
                    context_parts.append("Contact Information:")
                    contact = item.get('data', {})
                    for key, value in contact.items():
                        context_parts.append(f"  {key.title()}: {value}")
                
                elif item.get('type') == 'address':
                    context_parts.append(f"Address: {item.get('data', '')}")
                
                elif item.get('type') == 'appointment':
                    context_parts.append("Appointment Booking:")
                    appointment = item.get('data', {})
                    if 'booking_methods' in appointment:
                        context_parts.append("  Methods:")
                        for method in appointment['booking_methods']:
                            context_parts.append(f"    - {method}")
                    if 'emergency' in appointment:
                        context_parts.append(f"  Emergency: {appointment['emergency']}")
                
                elif item.get('type') == 'departments':
                    context_parts.append("Available Departments:")
                    departments = item.get('data', [])
                    context_parts.append(f"  {', '.join(departments)}")
                
                context_parts.append("")
        
        return "\n".join(context_parts).strip()
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], intent: str) -> str:
        """
        Generate response using the LLM.
        
        Args:
            query: User's query
            context: Retrieved context data
            intent: Classified intent
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Format context
        formatted_context = self.format_context(context, intent)
        
        # Create prompt
        prompt = self.current_prompt.format(
            retrieved_context=formatted_context,
            user_question=query,
            relevant_knowledge=formatted_context,
            user_query=query
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response
    
    def apply_safety_filters(self, response: str) -> str:
        """
        Apply safety filters to the response.
        
        Args:
            response: Generated response
            
        Returns:
            Filtered response
        """
        response_lower = response.lower()
        
        # Check for unsafe keywords
        for keyword in self.unsafe_keywords:
            if keyword in response_lower:
                return "I'm sorry, I cannot provide specific medical advice. Please consult a healthcare professional for personalized medical guidance."
        
        # Check for diagnostic language
        diagnostic_patterns = [
            r'\b(you have|you are|you should|you need to)\b',
            r'\b(diagnosis|diagnose|condition|disease)\b',
            r'\b(take this|use this|apply this)\b'
        ]
        
        for pattern in diagnostic_patterns:
            if re.search(pattern, response_lower):
                return "I'm sorry, I cannot provide specific medical advice. Please consult a healthcare professional for personalized medical guidance."
        
        return response
    
    def add_disclaimer(self, response: str) -> str:
        """
        Add educational disclaimer to the response.
        
        Args:
            response: Response text
            
        Returns:
            Response with disclaimer
        """
        disclaimer = "\n\nDisclaimer: This is for educational purposes only and not a substitute for professional medical advice."
        
        # Check if disclaimer already exists
        if "disclaimer" not in response.lower():
            response += disclaimer
        
        return response
    
    def generate_final_response(self, query: str, context: List[Dict[str, Any]], intent: str) -> str:
        """
        Complete response generation pipeline.
        
        Args:
            query: User's query
            context: Retrieved context data
            intent: Classified intent
            
        Returns:
            Final processed response
        """
        # Generate response
        response = self.generate_response(query, context, intent)
        
        # Apply safety filters
        response = self.apply_safety_filters(response)
        
        # Add disclaimer
        response = self.add_disclaimer(response)
        
        return response
    
    def switch_prompt_version(self, version: int):
        """
        Switch between prompt versions.
        
        Args:
            version: 1 or 2
        """
        if version == 1:
            self.current_prompt = self.promptv1
        elif version == 2:
            self.current_prompt = self.promptv2
        else:
            raise ValueError("Prompt version must be 1 or 2")
        
        print(f"Switched to prompt version {version}")
