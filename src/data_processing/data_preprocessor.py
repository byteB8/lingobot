"""
Data preprocessing module for healthcare chatbot.
Handles cleaning and structuring of FAQ, lifestyle tips, and hospital data.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path


class DataPreprocessor:
    """Handles preprocessing of healthcare data sources."""

    def __init__(self, data_dir: str = "data", raw_data_dir: str = "raw_data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = Path(raw_data_dir)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)

        return text

    def preprocess_faq_data(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Preprocess FAQ data from medquad.csv.
        Returns list of cleaned Q&A pairs.
        """
        df = pd.read_csv(csv_path)

        faq_data = []
        for _, row in df.iterrows():
            question = self.clean_text(str(row['question']))
            answer = self.clean_text(str(row['answer']))

            if question and answer:
                faq_data.append({
                    'question': question,
                    'answer': answer,
                    'source': str(row.get('source', 'Unknown')),
                    'focus_area': str(row.get('focus_area', 'General'))
                })

        return faq_data

    def preprocess_lifestyle_tips(self, csv_path: str) -> List[Dict[str, str]]:
        """
        Preprocess lifestyle tips from healthy-tips.csv.
        Returns list of cleaned tips with metadata.
        """
        df = pd.read_csv(csv_path)

        tips_data = []
        for _, row in df.iterrows():
            tip = self.clean_text(str(row['tips']))
            category = str(row.get('category', 'general'))
            source = str(row.get('source', 'Unknown'))

            if tip:
                tips_data.append({
                    'tip': tip,
                    'category': category,
                    'source': source
                })

        return tips_data

    def preprocess_hospital_data(self, json_path: str) -> Dict[str, Any]:
        """
        Preprocess hospital data from JSON file.
        Returns structured hospital information.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            hospital_data = json.load(f)

        # Clean and structure the data
        cleaned_data = {
            'basic_info': {
                'name': hospital_data['hospital']['name'],
                'address': hospital_data['hospital']['address'],
                'contact': hospital_data['hospital']['contact']
            },
            'working_hours': hospital_data['hospital']['working_hours'],
            'appointment': hospital_data['hospital']['appointment'],
            'departments': hospital_data['hospital']['departments'],
            'faq': []
        }

        # Clean FAQ data
        for faq in hospital_data['hospital']['faq']:
            cleaned_faq = {
                'question': self.clean_text(faq['question']),
                'answer': self.clean_text(faq['answer'])
            }
            cleaned_data['faq'].append(cleaned_faq)

        return cleaned_data

    def save_processed_data(self, data: Dict[str, Any], output_path: str) -> None:
        """Save processed data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def process_all_data(self) -> Dict[str, Any]:
        """
        Process all data sources and return structured data.
        """
        print("Processing FAQ data...")
        faq_data = self.preprocess_faq_data(self.raw_data_dir / "medquad.csv")

        print("Processing lifestyle tips...")
        tips_data = self.preprocess_lifestyle_tips(
            self.raw_data_dir / "healthy-tips.csv")

        print("Processing hospital data...")
        hospital_data = self.preprocess_hospital_data(
            self.data_dir / "lingo-hospital.json")

        processed_data = {
            'faq': faq_data,
            'lifestyle_tips': tips_data,
            'hospital': hospital_data,
            'metadata': {
                'faq_count': len(faq_data),
                'tips_count': len(tips_data),
                'hospital_faq_count': len(hospital_data['faq'])
            }
        }

        return processed_data


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_data()

    print(f"Processed {processed_data['metadata']['faq_count']} FAQ entries")
    print(
        f"Processed {processed_data['metadata']['tips_count']} lifestyle tips")
    print(
        f"Processed {processed_data['metadata']['hospital_faq_count']} hospital FAQ entries")

    # Save processed data
    preprocessor.save_processed_data(
        processed_data, "data/processed_data.json")
    print("Processed data saved to data/processed_data.json")
